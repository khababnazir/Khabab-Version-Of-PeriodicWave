"""
Stage 8 -- does a trained network respect the moire lattice's translation symmetry?

Run (on the CPU, under a minute for all four runs; it never touches a GPU,
so it is safe while a training run is going):
    python check_translation_symmetry.py

The Hamiltonian does not change when every electron is moved by the same
moire lattice vector a (the moire potential repeats with period a; kinetic
and Coulomb energy depend only on differences).  A wave function with that
symmetry satisfies |Psi(R + a)| = |Psi(R)| for every configuration R, so
log|Psi| would not change at all.  We move the 256 first walkers of the
final checkpoint by a1, a2 and a1 + a2 and print how much log|Psi| changes.

Control: moving ONE electron by a whole supercell vector must change nothing
(the network is built periodic); only float32 rounding remains.

Why it matters.  The evaluation found uneven site occupations (0.23 to 0.94
instead of 2/3 everywhere).  That could be the wave function, or walkers
that never mixed between translated copies of a pattern.  If log|Psi|
changes under a lattice translation, it is the wave function.
"""
import json
import os
import sys
from pathlib import Path

os.environ["JAX_PLATFORMS"] = "cpu"      # before JAX is imported: always off the GPUs

import numpy as np

HERE = Path(__file__).resolve().parent
RUNS = HERE.parents[3] / "runs" / "archive" / "stage08_authors_code" / "training"
sys.path.insert(0, str(HERE))
WALKERS = 256


def main():
    import jax
    import jax.numpy as jnp
    from periodicwave import checkpoint
    from evaluate_weights import training_configuration, build_like_train_py

    a1, a2 = np.array([1.0, 0.0]), np.array([-0.5, np.sqrt(3) / 2])        # moire lattice, a_M = 1
    for name in ("psiformer_eps10", "slaternet_eps10", "psiformer_eps5", "slaternet_eps5"):
        run = RUNS / name
        cfg, _ = training_configuration(run)
        network, _, _ = build_like_train_py(cfg)
        final = run / f"qmcjax_ckpt_{cfg.optim.iterations - 1:06d}.npz"
        _, data, params, _, _ = checkpoint.restore(str(final), cfg.batch_size)
        params = jax.tree_util.tree_map(lambda x: jnp.asarray(x)[0], params)
        x, spins, atoms, charges = (jnp.asarray(v)[0][:WALKERS]
                                    for v in (data.positions, data.spins, data.atoms, data.charges))
        log_abs_psi = jax.jit(jax.vmap(lambda xi, s, a, c: network.apply(params, xi, s, a, c)[1]))
        reference = log_abs_psi(x, spins, atoms, charges)

        def change_after_moving(shift, electrons=slice(None)):
            R = x.reshape(WALKERS, -1, 2).at[:, electrons, :].add(shift)
            return np.asarray(log_abs_psi(R.reshape(WALKERS, -1), spins, atoms, charges) - reference)

        supercell_vector = np.asarray(cfg.system.pbc_lattice)[:, 0]
        print(f"{name}")
        print(f"   control: electron 0 moved by a supercell vector   max |change| {np.abs(change_after_moving(supercell_vector, 0)).max():.1e}")
        for label, shift in (("a1", a1), ("a2", a2), ("a1 + a2", a1 + a2)):
            change = change_after_moving(shift)
            print(f"   all electrons moved by {label:8s} change of log|Psi|: mean {change.mean():+.3f}, "
                  f"spread {change.std():.3f}, max |change| {np.abs(change).max():.3f}")


if __name__ == "__main__":
    main()
