"""
Stage 7, part 2 -- the complete Hamiltonian, and a first interacting energy.

With ewald.py in place we can write the paper's Eq. (1) in full, in code
units (lengths in a_M, energies in hbar^2 / (m* a_M^2) = 3.3756 meV):

    H = - 1/2 sum_i grad_i^2                                   kinetic
        - 2 V sum_i sum_{n=1,2,3} cos(g_n . r_i + phi)         moire potential
        + U/2 sum_{i != j} 1/|r_i - r_j|   (Ewald, background, Madelung)

The first two lines are Stage 4, the third is this stage.  V = 4.443709
(15 meV); U = 10.623473 at eps = 5 and 5.311737 at eps = 10.  The dielectric
constant enters only through U.

Run:  python interacting_energy.py

It measures <H> for the SlaterNet trained in Stage 6.  That network is a
single Slater determinant optimised WITHOUT the Coulomb term, so for the full
H its energy is a variational upper bound for this 9-site system, though not
a tight one.

What to compare it with.  The paper's Fig. 5 is the same system (3 x 3
supercell, 6 electrons, filling 2/3).  No single determinant can go below
Hartree-Fock, the best single determinant, and the paper's SlaterNet is its
estimate of Hartree-Fock; ours was never optimised with Coulomb, so it should
come out above that.  Table I is NOT a valid comparison: it is for 27 sites,
and the energy per electron changes with system size, so it bounds nothing
here.

For a FIXED wave function <H_ee> is exactly proportional to U, so the two
eps values below differ in a trivial way.  They stop being trivially related
once the wave function is re-optimised for each eps -- that rearrangement of
the electrons is the physics of the paper (Stage 8).
"""
import json
import sys
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "stage03_slater"))
sys.path.insert(0, str(HERE.parent / "stage04_mcmc"))
from slaternet import Architecture, init_parameters, log_psi, supercell_vectors, reciprocal_vectors  # noqa: E402
from local_energy import moire_units, local_energy  # noqa: E402
from metropolis import run_metropolis  # noqa: E402
from ewald import make_ewald_setup, coulomb_energy, moire_interaction_strength  # noqa: E402

N_ELECTRONS = 6
N_SITES = 9
PHI = np.pi / 4
TRAINED_RUN = HERE.parent.parent / "runs" / "stage06" / "full"

# The paper's Fig. 5 (3 x 3 supercell, 6 electrons): energy per electron in
# meV, read off the plot by eye, so good to about +-0.05 meV.
FIG5 = {5.0: {"SlaterNet": -57.9, "self-attention": -59.5},
        10.0: {"SlaterNet": -31.6, "self-attention": -32.3}}


def local_energy_with_coulomb(log_psi_fn, positions, potential_strength, potential_phi,
                              ewald_setup, interaction_strength):
    """E_loc = (H Psi) / Psi for the full Hamiltonian, Eq. (1).

    The Coulomb term is an ordinary function of the positions, so it simply
    adds to Stage 4's local energy: no derivatives of Psi are involved.
    """
    kinetic_and_moire = local_energy(log_psi_fn, positions, potential_strength, potential_phi)
    return kinetic_and_moire + coulomb_energy(ewald_setup, positions, strength=interaction_strength)


def load_stage6_wave_function(key):
    """The SlaterNet trained in Stage 6, or a fresh one if that run is missing."""
    summary_file, params_file = TRAINED_RUN / "summary.json", TRAINED_RUN / "params.npz"
    if summary_file.exists() and params_file.exists():
        architecture = Architecture(**json.loads(summary_file.read_text())["architecture"])
        _, unflatten = ravel_pytree(init_parameters(key, architecture))
        flat = jnp.asarray(np.load(params_file)["flat_params"])
        return unflatten(flat), architecture, "trained in Stage 6, without Coulomb"
    architecture = Architecture(n_electrons=N_ELECTRONS, n_determinants=1, n_layers=2, width=32)
    return init_parameters(key, architecture), architecture, "UNTRAINED (Stage 6 run not found)"


def main():
    jax.config.update("jax_enable_x64", True)
    energy_unit_meV, potential_strength = moire_units()
    lattice = supercell_vectors(N_SITES)
    G = reciprocal_vectors(lattice)
    setup = make_ewald_setup(lattice)
    key = jax.random.key(0)

    params, architecture, provenance = load_stage6_wave_function(key)
    print(f"wave function: SlaterNet, {architecture.n_layers} layers of width "
          f"{architecture.width}, {provenance}")

    def psi(positions):
        return log_psi(params, positions, G)

    @jax.jit
    def log_abs_batch(walkers):
        return jax.vmap(lambda R: psi(R).log_abs)(walkers)

    def mean_energy_function(U):
        @jax.jit
        def mean_energy(walkers):
            e_loc = jax.vmap(lambda R: local_energy_with_coulomb(
                psi, R, potential_strength, PHI, setup, U))(walkers)
            return jnp.mean(jnp.real(e_loc))
        return mean_energy

    dielectric_constants = (5.0, 10.0)
    interaction = {eps: moire_interaction_strength(dielectric_constant=eps)
                   for eps in dielectric_constants}
    mean_energy = {eps: mean_energy_function(interaction[eps]) for eps in dielectric_constants}

    # Sample |Psi|^2: burn in, then measure in 20 rounds with the step width frozen.
    # The wave function is fixed, so both eps values are measured on the same walkers.
    key, key_start, key_burn = jax.random.split(key, 3)
    walkers = jax.random.uniform(key_start, (1024, N_ELECTRONS, 2)) @ jnp.asarray(lattice).T
    walkers, width, _ = run_metropolis(log_abs_batch, walkers, key_burn, n_steps=300, step_width=0.2)

    rounds = {eps: [] for eps in dielectric_constants}
    for _ in range(20):
        key, key_round = jax.random.split(key)
        walkers, width, _ = run_metropolis(log_abs_batch, walkers, key_round,
                                           n_steps=30, step_width=width, adapt_every=None)
        for eps in dielectric_constants:
            rounds[eps].append(float(mean_energy[eps](walkers)))

    print(f"\n{N_ELECTRONS} electrons on {N_SITES} moire sites (filling 2/3), "
          f"1024 walkers, 20 rounds of 30 steps")
    print("\n  eps   U           <H>/N (meV)         paper's Fig. 5, same system (meV)")
    print("                                        SlaterNet   self-attention")
    for eps in dielectric_constants:
        per_electron_meV = np.asarray(rounds[eps]) / N_ELECTRONS * energy_unit_meV
        mean = per_electron_meV.mean()
        error = per_electron_meV.std(ddof=1) / np.sqrt(len(per_electron_meV))
        print(f"  {eps:4.1f}  {interaction[eps]:9.6f}   {mean:+9.3f} +- {error:5.3f}     "
              f"{FIG5[eps]['SlaterNet']:+7.1f}     {FIG5[eps]['self-attention']:+7.1f}")

    print("\n  Ours should lie above the paper's SlaterNet: both are single determinants")
    print("  for the same Hamiltonian, and only theirs was optimised for it.")
    print("  Error bars treat the 20 rounds as independent; blocking comes in Stage 8.")


if __name__ == "__main__":
    main()
