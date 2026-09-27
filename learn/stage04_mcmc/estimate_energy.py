"""
Stage 4, part 3 -- put it together: estimate the energy of a wave function.

Run:  python estimate_energy.py

Two runs:

  A. A Slater determinant of plane waves, with NO potential.  This is an
     exact eigenstate of the kinetic energy, so every sampled configuration
     must give the same local energy: zero variance (paper Eq. 28).  If the
     sampler or the kinetic energy were wrong, this would show it.

  B. SlaterNet from Stage 3, freshly initialised, in the moire potential.
     Not an eigenstate, so E_loc fluctuates from configuration to
     configuration, and its average is the energy of this untrained wave
     function.  Stage 6 will lower that energy by training.
"""
import sys
import time
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)    # float64 so run A can show a variance of ~0
import numpy as np
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "stage03_slater"))
from slaternet import Architecture, init_parameters, log_psi, supercell_vectors, reciprocal_vectors  # noqa: E402
from local_energy import (moire_units, local_energy, plane_wave_k_vectors,  # noqa: E402
                          plane_wave_log_psi)
from metropolis import run_metropolis  # noqa: E402


N_ELECTRONS = 6            # 9 moire sites at filling 2/3
N_SITES = 9
N_WALKERS = 512
BURN_IN_STEPS = 300        # the authors' cfg.mcmc.burn_in
STEPS_BETWEEN = 30         # the authors' cfg.mcmc.steps
N_MEASUREMENTS = 10
PHI = np.pi / 4            # moire shape angle, 45 degrees


def sample_and_measure(log_psi_fn, potential_strength, key):
    """Burn in, then alternate: move the walkers, measure E_loc on all of them.

    Returns an array of shape (N_MEASUREMENTS, N_WALKERS) of complex local energies.
    """
    lattice = supercell_vectors(N_SITES)

    # log|Psi| for a whole batch of walkers, compiled once.
    def log_abs_one(R):
        return log_psi_fn(R).log_abs
    log_abs_batch = jax.jit(jax.vmap(log_abs_one))

    # E_loc for a whole batch of walkers, compiled once.
    def e_loc_one(R):
        return local_energy(log_psi_fn, R, potential_strength, PHI)
    e_loc_batch = jax.jit(jax.vmap(e_loc_one))

    # Start the walkers uniformly inside the supercell.
    key, key_init = jax.random.split(key)
    fractional = jax.random.uniform(key_init, (N_WALKERS, N_ELECTRONS, 2))
    walkers = fractional @ jnp.asarray(lattice).T

    # Burn-in: let the walkers forget where they started.
    key, key_burn = jax.random.split(key)
    walkers, width, rates = run_metropolis(log_abs_batch, walkers, key_burn,
                                           BURN_IN_STEPS, step_width=0.2)
    print(f"  burn-in done: step width {width:.3f}, acceptance over last 30 steps "
          f"{np.mean(rates[-30:]):.2f}")

    measurements = []
    for m in range(N_MEASUREMENTS):
        key, key_m = jax.random.split(key)
        walkers, width, _ = run_metropolis(log_abs_batch, walkers, key_m,
                                           STEPS_BETWEEN, step_width=width)
        measurements.append(np.asarray(e_loc_batch(walkers)))
    return np.array(measurements)


def report(e_loc, energy_unit_meV):
    """Print the energy estimate from an array (N_MEASUREMENTS, N_WALKERS) of E_loc.

    Error bar: the standard error of the mean over the measurement rounds.
    Rounds 30 steps apart are not perfectly independent, so this slightly
    underestimates the true error.  (The paper's error bars are the spread of
    the batch-averaged energy between optimisation steps -- the same idea.)
    """
    n_rounds = e_loc.shape[0]

    # Real part: the energy.
    round_means = e_loc.real.mean(axis=1)                    # one number per round
    energy = round_means.mean()
    std_error = round_means.std(ddof=1) / np.sqrt(n_rounds)

    # Imaginary part: must average to zero.  Give it the same kind of error
    # bar, so "is it zero?" becomes a question with an answer.
    imag_round_means = e_loc.imag.mean(axis=1)
    mean_imag = imag_round_means.mean()
    imag_std_error = imag_round_means.std(ddof=1) / np.sqrt(n_rounds)

    variance = e_loc.real.var()

    print(f"  energy         = {energy:+.6f} +- {std_error:.6f}   "
          f"({energy * energy_unit_meV:+.4f} +- {std_error * energy_unit_meV:.4f} meV)")
    print(f"  per electron   = {energy * energy_unit_meV / N_ELECTRONS:+.4f} meV")
    print(f"  variance E_loc = {variance:.3e}")
    print(f"  mean Im E_loc  = {mean_imag:+.3e} +- {imag_std_error:.3e}   (should be 0 within the error)")
    return energy, std_error


if __name__ == "__main__":
    energy_unit_meV, potential_strength = moire_units()
    lattice = supercell_vectors(N_SITES)
    G = reciprocal_vectors(lattice)
    print(f"energy unit = {energy_unit_meV:.6f} meV,  moire potential strength = {potential_strength:.6f}\n")

    # ---------------- A: plane waves, no potential: an exact eigenstate ----------------
    t0 = time.perf_counter()
    k = plane_wave_k_vectors(lattice, N_ELECTRONS)
    exact = 0.5 * np.sum(k**2)
    print(f"A. plane-wave determinant, no potential  (exact energy {exact:.6f})")
    e_loc_A = sample_and_measure(lambda R: plane_wave_log_psi(R, k),
                                 potential_strength=0.0, key=jax.random.key(0))
    report(e_loc_A, energy_unit_meV)
    print(f"  largest |E_loc - exact| over all samples = {np.max(np.abs(e_loc_A - exact)):.2e}")
    print(f"  ({time.perf_counter() - t0:.0f} s)\n")

    # ---------------- B: untrained SlaterNet in the moire potential ----------------
    t0 = time.perf_counter()
    params = init_parameters(jax.random.key(1), Architecture(n_electrons=N_ELECTRONS))
    print("B. untrained SlaterNet (1 determinant) in the moire potential")
    e_loc_B = sample_and_measure(lambda R: log_psi(params, R, G),
                                 potential_strength=potential_strength, key=jax.random.key(2))
    report(e_loc_B, energy_unit_meV)
    print(f"  ({time.perf_counter() - t0:.0f} s)")
