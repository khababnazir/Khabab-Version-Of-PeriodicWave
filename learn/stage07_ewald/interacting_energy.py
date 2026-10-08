"""
Stage 7, part 2 -- the complete Hamiltonian, and a first interacting energy.

With ewald.py in place we can finally write Eq. (1) of the paper in full, in
code units (lengths in a_M, energies in hbar^2 / m* a_M^2):

    H = - 1/2 sum_i grad_i^2                                   kinetic
        - 2 V sum_i sum_{n=1,2,3} cos(g_n . r_i + phi)         moire potential
        + U/2 sum_{i != j} 1/|r_i - r_j|   (Ewald, + background + Madelung)

The first two lines are Stage 4; the third is this stage.  V = 4.443709 for
the paper's 15 meV potential; U = 10.623473 at eps = 5 and 5.311737 at
eps = 10 -- the interaction strength is the ONLY thing that changes between
the paper's two cases.

Run:  python interacting_energy.py

It measures <H> for the SlaterNet that Stage 6 trained.  That wave function
was optimised for the non-interacting Hamiltonian, so the number it gives is
an honest variational upper bound for the interacting one, but a poor one --
improving it is Stage 8.  One thing worth noticing in the output: for a
FIXED wave function <H_ee> is exactly proportional to U, so eps = 5 and
eps = 10 differ trivially here.  They stop being trivially related as soon
as the wave function is re-optimised for each, because the electrons
rearrange.  That rearrangement is the physics of the paper.
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
sys.path.insert(0, str(HERE.parent / "stage06_natural_gradient"))
from slaternet import Architecture, init_parameters, log_psi, supercell_vectors, reciprocal_vectors  # noqa: E402
from local_energy import moire_units, local_energy  # noqa: E402
from metropolis import run_metropolis  # noqa: E402
from exact_noninteracting import exact_ground_state_energy  # noqa: E402
from ewald import make_ewald_setup, coulomb_energy, moire_interaction_strength  # noqa: E402

N_ELECTRONS = 6
N_SITES = 9
PHI = np.pi / 4
TRAINED_RUN = HERE.parent.parent / "runs" / "stage06" / "full"


def local_energy_with_coulomb(log_psi_fn, positions, potential_strength, potential_phi,
                              ewald_setup, interaction_strength):
    """E_loc = (H Psi)/Psi for the full Hamiltonian.

    The Coulomb term is a plain function of the positions, so it just adds
    to the local energy -- no derivatives of Psi are involved in it.
    """
    non_interacting = local_energy(log_psi_fn, positions, potential_strength, potential_phi)
    return non_interacting + coulomb_energy(ewald_setup, positions, strength=interaction_strength)


def load_stage6_wave_function(key):
    """The SlaterNet trained in Stage 6, or a fresh one if that run is missing."""
    summary_file, params_file = TRAINED_RUN / "summary.json", TRAINED_RUN / "params.npz"
    if summary_file.exists() and params_file.exists():
        architecture = Architecture(**json.loads(summary_file.read_text())["architecture"])
        _, unflatten = ravel_pytree(init_parameters(key, architecture))
        flat = jnp.asarray(np.load(params_file)["flat_params"])
        return unflatten(flat), architecture, "trained in Stage 6"
    architecture = Architecture(n_electrons=N_ELECTRONS, n_determinants=1, n_layers=2, width=32)
    return init_parameters(key, architecture), architecture, "untrained (Stage 6 run not found)"


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

    @jax.jit
    def log_abs_batch(walkers):
        return jax.vmap(lambda R: log_psi(params, R, G).log_abs)(walkers)

    @jax.jit
    def pieces(walkers):
        """Both halves of H at every walker, kept apart so we can see each."""
        kinetic_and_moire = jax.vmap(
            lambda R: local_energy(lambda X: log_psi(params, X, G), R, potential_strength, PHI))(walkers)
        bare_coulomb = jax.vmap(lambda R: coulomb_energy(setup, R))(walkers)
        return jnp.real(kinetic_and_moire), bare_coulomb

    # Sample |Psi|^2: burn in, then measure in rounds with the step width frozen.
    key, key_start, key_burn = jax.random.split(key, 3)
    walkers = jax.random.uniform(key_start, (1024, N_ELECTRONS, 2)) @ jnp.asarray(lattice).T
    walkers, width, _ = run_metropolis(log_abs_batch, walkers, key_burn, n_steps=300, step_width=0.2)

    non_interacting_rounds, coulomb_rounds = [], []
    for _ in range(20):
        key, key_round = jax.random.split(key)
        walkers, width, _ = run_metropolis(log_abs_batch, walkers, key_round,
                                           n_steps=30, step_width=width, adapt_every=None)
        kinetic_and_moire, bare_coulomb = pieces(walkers)
        non_interacting_rounds.append(float(jnp.mean(kinetic_and_moire)))
        coulomb_rounds.append(float(jnp.mean(bare_coulomb)))

    def mean_and_error(rounds):
        rounds = np.asarray(rounds)
        return rounds.mean(), rounds.std(ddof=1) / np.sqrt(len(rounds))

    non_interacting, non_interacting_error = mean_and_error(non_interacting_rounds)
    bare_coulomb, bare_coulomb_error = mean_and_error(coulomb_rounds)
    exact_non_interacting, _ = exact_ground_state_energy(lattice, N_ELECTRONS, potential_strength, PHI)

    print(f"\n{N_ELECTRONS} electrons on {N_SITES} moire sites (filling 2/3), "
          f"1024 walkers, 20 rounds")
    print(f"  kinetic + moire        {non_interacting:+.5f} +- {non_interacting_error:.5f} code units"
          f"   (exact without Coulomb: {exact_non_interacting:+.5f})")
    print(f"  Coulomb sum, U = 1     {bare_coulomb:+.5f} +- {bare_coulomb_error:.5f}")
    # The paper reports energy PER ELECTRON in meV: Figs. 3, 5 and 9 are all
    # labelled E/N (meV), and Table I is the same comparison as Fig. 5.  Table I
    # is for 27 sites, not our 9, so the last column is a scale check, not a target.
    print("\n  eps      U        <H>/N (code units)    <H>/N (meV)    Table I, 27 sites (meV)")
    table_one = {5.0: -59.127, 10.0: -32.070}
    for dielectric in (5.0, 10.0):
        U = moire_interaction_strength(dielectric_constant=dielectric)
        total = (non_interacting + U * bare_coulomb) / N_ELECTRONS
        error = np.hypot(non_interacting_error, U * bare_coulomb_error) / N_ELECTRONS
        print(f"  {dielectric:4.1f}  {U:8.5f}   {total:+9.4f} +- {error:6.4f}   "
              f"{total * energy_unit_meV:+9.3f}      {table_one[dielectric]:+9.3f}")

    print("\n  Ours should sit slightly ABOVE the paper's: this wave function was")
    print("  optimised for the non-interacting problem, and 9 sites is not 27.")
    print("  Error bars ignore correlations between rounds; blocking comes in Stage 8.")


if __name__ == "__main__":
    main()
