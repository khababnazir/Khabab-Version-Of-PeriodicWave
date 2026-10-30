"""Do the fast and the reference local energy agree in the precision of a real run?  (Stage 8)

test_train_moire.py compares them in double precision, with the attention's
softmax written out exactly.  A training run uses single precision on the GPU
and JAX's own attention, whose softmax is single precision.  This script
compares, for the Table II attention network at 18 electrons with its initial
weights (seed 0, as the runs start), at walkers after a short burn-in:

    reference   full Hessian (Stage 4) + det (Stage 3)
    fast        folx + slogdet

It prints the largest difference of single local energies, and the
difference of the two batch means next to the batch's statistical error
(meV per electron), as one JSON line.

Usage:  CUDA_VISIBLE_DEVICES=0 python production_check.py [WALKERS]      (default 512; one GPU)
"""
import dataclasses
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import train_moire as tm  # noqa: E402

gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "")
if not gpu.isdigit():
    sys.exit("make exactly one GPU visible, e.g. CUDA_VISIBLE_DEVICES=0")
lock = tm.hold_lock(tm.GPU_LOCKS / f"gpu{gpu}.lock", f"GPU {gpu} is in use by another job")  # noqa: F841
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
assert jax.default_backend() == "gpu", jax.default_backend()
jax.config.update("jax_default_matmul_precision", "float32")      # as train_moire.py

n_walkers = int(sys.argv[1]) if len(sys.argv) > 1 else 512
MEV_PER_ELECTRON = 3.375559 / 18                                    # code unit of energy, per electron
fast = dataclasses.replace(tm.TABLE_II, batch=n_walkers, laplacian="folx", determinants="slogdet")
reference = dataclasses.replace(fast, laplacian="hessian", determinants="det")
problems = {"reference": tm.make_problem("attention", 5.0, 18, reference),
            "fast": tm.make_problem("attention", 5.0, 18, fast)}

# The start of a run (train_moire.train): the same keys, so the same weights and walkers.
key = jax.random.PRNGKey(0)
key, key_params, key_walkers = jax.random.split(key, 3)
params = problems["fast"].init(key_params, problems["fast"].arch)
walkers = fast.init_width * jax.random.normal(key_walkers, (n_walkers, 18, 2))
move, _ = tm.make_sampler(problems["fast"], fast, "real")
for _ in range(100):                                                # 100 x 30 Metropolis steps
    key, key_move = jax.random.split(key)
    walkers, pmove = move(params, walkers, key_move, fast.move_width)

energies = {}
for name, problem in problems.items():
    local_energies = jax.jit(jax.vmap(lambda R, problem=problem: problem.local_energy(params, R)))
    energies[name] = np.asarray(local_energies(walkers))

ref, new = energies["reference"], energies["fast"]
difference = new - ref
print(json.dumps(dict(
    walkers=n_walkers, last_acceptance=round(float(pmove), 3),
    not_finite=dict(reference=int(np.sum(~np.isfinite(ref))), fast=int(np.sum(~np.isfinite(new)))),
    max_difference_over_max_value=dict(
        real=float(np.max(np.abs(difference.real)) / np.max(np.abs(ref.real))),
        imag=float(np.max(np.abs(difference.imag)) / np.max(np.abs(ref.imag)))),
    mean_real_meV_per_electron=dict(reference=float(ref.real.mean() * MEV_PER_ELECTRON),
                                    fast=float(new.real.mean() * MEV_PER_ELECTRON)),
    mean_difference_meV_per_electron=float(difference.real.mean() * MEV_PER_ELECTRON),
    statistical_error_meV_per_electron=float(ref.real.std() / np.sqrt(n_walkers) * MEV_PER_ELECTRON))))
