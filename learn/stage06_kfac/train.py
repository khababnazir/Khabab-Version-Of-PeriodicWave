"""
Stage 6, part 3 -- train a wave function: variational Monte Carlo.

Run:  python train.py --estimator full        --out ../../runs/stage06/full
      python train.py --estimator real_only   --out ../../runs/stage06/real_only
      (add --seed 1 for a second, independent run)

Writes to --out:  history.csv (every iteration), summary.json (final frozen
measurement, settings, architecture, versions) and params.npz (the trained
parameters and final walkers, so the wave function can be reloaded).

The loop, once per iteration:
  1. move the walkers (Metropolis, Stage 4) so they follow the CURRENT |Psi|^2
  2. local energy E_loc at every walker (Stage 4)
  3. log-derivatives O = d log Psi / d theta at every walker (gradient.py)
  4. gradient from E_loc and O, then a natural-gradient step using S
  5. update the parameters, record the energy

The problem: 6 electrons on 9 moire sites, kinetic energy + moire potential,
no Coulomb yet (Stage 7).  Its exact answer is known (exact_noninteracting.py),
so we can see how close training gets.  The exact energy of any trial Psi
cannot go below it (variational principle), but a Monte Carlo ESTIMATE can
dip below by statistical noise.  The plane-wave answer itself is converged
to about 2e-9 code units (cutoff 30 versus 60).

The experiment: --estimator full uses the exact gradient formula;
--estimator real_only drops Im E_loc, as the authors' folx path does.
Everything else is identical, including the random seed.
"""
import argparse
import csv
import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np
import jax
# Full float32 accuracy in matrix products.  On Ampere GPUs (the RTX 3090) JAX
# otherwise uses TF32, which keeps only ~3 significant digits in products.
# That made S visibly non-positive (eigenvalues down to -12).  The authors'
# run script sets exactly this, for the same reason.
jax.config.update("jax_default_matmul_precision", "float32")
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "stage03_slater"))
sys.path.insert(0, str(HERE.parent / "stage04_mcmc"))
from slaternet import Architecture, init_parameters, log_psi, supercell_vectors, reciprocal_vectors  # noqa: E402
from local_energy import moire_units, local_energy  # noqa: E402
from metropolis import run_metropolis  # noqa: E402
from gradient import (log_derivatives, energy_gradient, quantum_geometric_tensor,  # noqa: E402
                      clip_local_energy, natural_gradient_step)
from exact_noninteracting import exact_ground_state_energy  # noqa: E402

N_ELECTRONS = 6
N_SITES = 9
PHI = np.pi / 4


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--estimator", choices=["full", "real_only"], default="full")
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--walkers", type=int, default=1024)
    parser.add_argument("--mcmc_steps", type=int, default=10, help="Metropolis steps per iteration")
    parser.add_argument("--learning_rate", type=float, default=0.05)
    parser.add_argument("--lr_delay", type=float, default=500.0, help="t0 in lr0 / (1 + t/t0)")
    parser.add_argument("--damping", type=float, default=1e-3)
    parser.add_argument("--max_step_norm", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------------- the physical system ----------------
    energy_unit_meV, strength = moire_units()
    lattice = supercell_vectors(N_SITES)
    G = reciprocal_vectors(lattice)
    exact_energy, _ = exact_ground_state_energy(lattice, N_ELECTRONS, strength, PHI)

    # ---------------- the wave function ----------------
    # A small SlaterNet, so that S (n_params x n_params) is cheap to build and solve.
    # One determinant is enough: the non-interacting ground state IS one determinant.
    arch = Architecture(n_electrons=N_ELECTRONS, n_determinants=1, n_layers=2, width=32)
    params = init_parameters(jax.random.key(args.seed), arch)
    n_params = ravel_pytree(params)[0].size
    print(f"estimator={args.estimator}  parameters={n_params}  walkers={args.walkers}  "
          f"device={jax.devices()[0].device_kind}")
    print(f"exact non-interacting energy = {exact_energy:+.6f}")

    # ---------------- compiled building blocks ----------------
    # Each takes params as an argument, so it is compiled once and reused as params change.
    @jax.jit
    def log_abs_batch(params, walkers):
        return jax.vmap(lambda R: log_psi(params, R, G).log_abs)(walkers)

    @jax.jit
    def e_loc_batch(params, walkers):
        def one(R):
            return local_energy(lambda X: log_psi(params, X, G), R, strength, PHI)
        return jax.vmap(one)(walkers)

    @jax.jit
    def O_batch(params, walkers):
        return log_derivatives(lambda p, R: log_psi(p, R, G), params, walkers)

    @jax.jit
    def update(params, e_loc, O, learning_rate):
        e_clipped = clip_local_energy(e_loc)
        if args.estimator == "real_only":
            e_clipped = jnp.real(e_clipped) + 0j            # the authors' folx path
        gradient = energy_gradient(e_clipped, O)
        S = quantum_geometric_tensor(O)
        step = natural_gradient_step(gradient, S, learning_rate, args.damping, args.max_step_norm)
        flat, unflatten = ravel_pytree(params)
        return unflatten(flat + step)

    # ---------------- walkers: start uniform, burn in ----------------
    key = jax.random.key(args.seed + 1)
    key, key_init, key_burn = jax.random.split(key, 3)
    walkers = jax.random.uniform(key_init, (args.walkers, N_ELECTRONS, 2)) @ jnp.asarray(lattice).T
    walkers, width, _ = run_metropolis(lambda W: log_abs_batch(params, W), walkers, key_burn,
                                       n_steps=200, step_width=0.2)

    # ---------------- training ----------------
    history_file = open(out_dir / "history.csv", "w", newline="")
    history = csv.writer(history_file)
    history.writerow(["iteration", "energy", "energy_error", "variance", "acceptance", "step_width", "learning_rate"])

    t_start = time.perf_counter()
    for t in range(args.iterations):
        # 1. walkers follow the current |Psi|^2.  Psi changes every iteration, so the
        #    step width keeps adapting during training, as the authors do.
        key, key_mcmc = jax.random.split(key)
        walkers, width, rates = run_metropolis(lambda W: log_abs_batch(params, W), walkers, key_mcmc,
                                               n_steps=args.mcmc_steps, step_width=width,
                                               adapt_every=args.mcmc_steps)

        # 2-3. local energies and log-derivatives at the new walkers
        e_loc = e_loc_batch(params, walkers)
        O = O_batch(params, walkers)

        # 4-5. natural-gradient step, with the paper's learning-rate schedule lr0 / (1 + t/t0)
        learning_rate = args.learning_rate / (1.0 + t / args.lr_delay)
        params = update(params, e_loc, O, learning_rate)

        # record (unclipped energies)
        energy = float(jnp.mean(jnp.real(e_loc)))
        variance = float(jnp.mean(jnp.abs(e_loc - jnp.mean(e_loc))**2))
        energy_error = float(jnp.std(jnp.real(e_loc)) / np.sqrt(args.walkers))
        history.writerow([t, energy, energy_error, variance, float(np.mean(rates)), width, learning_rate])

        if t % 50 == 0 or t == args.iterations - 1:
            history_file.flush()
            print(f"  it {t:5d}   E = {energy:+.5f} +- {energy_error:.5f}   var = {variance:.3e}   "
                  f"accept = {np.mean(rates):.2f}   ({time.perf_counter() - t_start:.0f} s)", flush=True)
    history_file.close()

    # ---------------- final measurement with the parameters frozen ----------------
    key, key_burn, key_measure = jax.random.split(key, 3)
    walkers, width, _ = run_metropolis(lambda W: log_abs_batch(params, W), walkers, key_burn,
                                       n_steps=100, step_width=width)
    round_means, all_e = [], []
    for _ in range(20):
        key_measure, key_round = jax.random.split(key_measure)
        walkers, width, _ = run_metropolis(lambda W: log_abs_batch(params, W), walkers, key_round,
                                           n_steps=30, step_width=width, adapt_every=None)
        e_loc = np.asarray(e_loc_batch(params, walkers))
        round_means.append(e_loc.real.mean())
        all_e.append(e_loc)
    all_e = np.concatenate(all_e)
    final_energy = float(np.mean(round_means))
    final_error = float(np.std(round_means, ddof=1) / np.sqrt(len(round_means)))   # approximate, see Stage 4
    final_variance = float(np.mean(np.abs(all_e - all_e.mean())**2))

    # ---------------- save the trained wave function ----------------
    # To reload:  arch = Architecture(**summary["architecture"])
    #             _, unflatten = ravel_pytree(init_parameters(jax.random.key(0), arch))
    #             params = unflatten(jnp.asarray(np.load("params.npz")["flat_params"]))
    flat_params, _ = ravel_pytree(params)
    np.savez(out_dir / "params.npz", flat_params=np.asarray(flat_params), walkers=np.asarray(walkers))

    summary = {
        "estimator": args.estimator,
        "exact_energy": exact_energy,
        "final_energy": final_energy,
        "final_error": final_error,
        "final_variance": final_variance,
        "energy_above_exact": final_energy - exact_energy,
        "energy_above_exact_meV": (final_energy - exact_energy) * energy_unit_meV,
        "n_params": int(n_params),
        "seconds": time.perf_counter() - t_start,
        "settings": vars(args),
        "architecture": dataclasses.asdict(arch),
        "jax_version": jax.__version__,
        "device": jax.devices()[0].device_kind,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"final: E = {final_energy:+.6f} +- {final_error:.6f}   exact {exact_energy:+.6f}   "
          f"above exact by {final_energy - exact_energy:+.2e}   variance {final_variance:.3e}")


if __name__ == "__main__":
    main()
