"""
Stage 8 checks for observables.py.  Run:  JAX_PLATFORMS=cpu python test_observables.py   (under a minute)

Each test guards a mistake that would silently change what the figures say:
  sites         the 9 sites must be the minima of the authors' moire potential,
                or "the electron's nearest site" means nothing;
  images        the minimum-image distance must be the true shortest one,
                or g(r) is wrong near the edge of the cell;
  honeycombs    exactly three patterns count as the crystal, and they must be
                the three lowest-Coulomb arrangements of Stage 7;
  uncorrelated  independent, uniformly spread electrons must give n = N/A,
                g = 1 and S(q) = 1 -- this fixes every normalisation;
  crystal       electrons fixed on a honeycomb must give the S(q) worked out
                by hand: 1.5 at the corners kappa of the moire Brillouin zone,
                6 at the moire reciprocal vectors.
"""
import itertools
import sys
from pathlib import Path

from absl.testing import absltest
import numpy as np
import jax
import jax.numpy as jnp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "stage07_ewald"))
import observables  # noqa: E402
from ewald import make_ewald_setup, coulomb_energy, moire_interaction_strength  # noqa: E402
from periodicwave.pbc import hamiltonian, lattices  # noqa: E402

SUPERCELL, MOIRE_LATTICE = lattices._triangular_lattice_vecs_periodic_potential(1.0, 9)
N_ELECTRONS = 6
KAPPA = 4 * np.pi / 3                  # |corner of the moire Brillouin zone|, a_M = 1
G = 4 * np.pi / np.sqrt(3)             # |moire reciprocal lattice vector|


class ObservablesTest(absltest.TestCase):

    def test_sites_are_the_minima_of_the_moire_potential(self):
        potential = hamiltonian.make_triangular_potential(
            MOIRE_LATTICE, jnp.array([0.0]), jnp.array([0.0]), 4.443709, np.pi / 4)
        one_electron = jax.vmap(lambda r: potential(r[None, :]))
        grid = (np.stack(np.meshgrid(np.arange(180), np.arange(180), indexing="ij"), -1)
                .reshape(-1, 2) + 0.5) / 180
        on_grid = one_electron(jnp.asarray(grid @ SUPERCELL.T))
        at_sites = one_electron(jnp.asarray(observables.moire_sites(SUPERCELL)))
        np.testing.assert_allclose(at_sites, at_sites[0], rtol=1e-6)      # all 9 sites equivalent
        self.assertLessEqual(float(at_sites.max()), float(on_grid.min()) + 1e-4)

    def test_minimum_image_is_the_shortest_image(self):
        d = np.random.default_rng(0).uniform(-6, 6, size=(5000, 2)).astype(np.float32)
        ours = np.linalg.norm(observables.minimum_image(jnp.asarray(d), SUPERCELL), axis=-1)
        images = np.array(list(itertools.product(range(-4, 5), repeat=2))) @ SUPERCELL.T
        shortest = np.min(np.linalg.norm(d[:, None, :] + images, axis=-1), axis=1)
        np.testing.assert_allclose(ours, shortest, rtol=1e-5, atol=1e-5)

    def test_honeycombs_are_the_three_lowest_coulomb_arrangements(self):
        sites = observables.moire_sites(SUPERCELL)
        setup = make_ewald_setup(SUPERCELL)
        strength = moire_interaction_strength(dielectric_constant=5.0)
        energy = {}
        for filled in itertools.combinations(range(observables.N_SITES), N_ELECTRONS):
            code = sum(2 ** k for k in filled)
            energy[code] = float(coulomb_energy(setup, jnp.asarray(sites[list(filled)]), strength=strength))
        by_energy = sorted(energy, key=energy.get)
        self.assertCountEqual(observables.honeycomb_codes(SUPERCELL), by_energy[:3])
        self.assertGreater(energy[by_energy[3]] - energy[by_energy[2]], 1.0)    # clearly separated

    def test_uncorrelated_electrons_give_flat_maps_and_unit_structure_factor(self):
        walkers = 20000
        s = np.random.default_rng(1).uniform(size=(walkers, N_ELECTRONS, 2))
        R = jnp.asarray(s @ SUPERCELL.T, dtype=jnp.float32)
        qs, _ = observables.supercell_q_vectors(SUPERCELL, q_max=8.0)
        sums = jax.device_get(observables.measure(R, SUPERCELL, qs, bins=10, r_max=1.5, n_radial_bins=30))

        self.assertEqual(sums["density"].sum(), walkers * N_ELECTRONS)            # each electron once
        self.assertEqual(sums["patterns"].sum(), walkers)                         # each configuration once
        self.assertEqual(sums["occupation"].sum(), walkers * N_ELECTRONS)
        n = observables.density_map(sums["density"], walkers, SUPERCELL)
        np.testing.assert_allclose(n.mean(), N_ELECTRONS / observables.supercell_area(SUPERCELL), rtol=1e-12)
        np.testing.assert_allclose(n, N_ELECTRONS / observables.supercell_area(SUPERCELL), rtol=0.15)

        g_map = observables.pair_correlation_map(sums["pairs"], walkers, N_ELECTRONS, SUPERCELL)
        np.testing.assert_allclose(g_map.mean(), 1.0, rtol=1e-12)
        np.testing.assert_allclose(g_map, 1.0, rtol=0.07)
        r, g_r = observables.radial_pair_correlation(sums["pair_distances"], walkers, N_ELECTRONS,
                                                     SUPERCELL, r_max=1.5)
        np.testing.assert_allclose(g_r[r > 0.3], 1.0, rtol=0.1)

        S, connected = observables.structure_factor(sums["rho_q"], sums["rho_q_squared"], walkers, N_ELECTRONS)
        np.testing.assert_allclose(S, 1.0, rtol=0.05)
        np.testing.assert_allclose(connected, 1.0, rtol=0.05)

    def test_a_honeycomb_crystal_has_the_structure_factor_worked_out_by_hand(self):
        code = observables.honeycomb_codes(SUPERCELL)[0]
        filled = [k for k in range(observables.N_SITES) if code >> k & 1]
        R = jnp.asarray(observables.moire_sites(SUPERCELL)[filled][None], dtype=jnp.float32)
        qs, _ = observables.supercell_q_vectors(SUPERCELL, q_max=8.0)
        sums = jax.device_get(observables.measure(R, SUPERCELL, qs, bins=30, r_max=1.5, n_radial_bins=30))
        S, connected = observables.structure_factor(sums["rho_q"], sums["rho_q_squared"], 1, N_ELECTRONS)

        q = np.linalg.norm(qs, axis=1)
        at_kappa, at_g = np.isclose(q, KAPPA, atol=1e-6), np.isclose(q, G, atol=1e-6)
        self.assertEqual((at_kappa.sum(), at_g.sum()), (6, 6))
        # rho_kappa = (all 9 sites, which sum to 0) - (the 3 empty sites, all in phase): |rho|^2 = 9.
        np.testing.assert_allclose(S[at_kappa], 9 / N_ELECTRONS, rtol=1e-5)
        # every electron exactly on a moire site: all in phase at g, |rho|^2 = 36.
        np.testing.assert_allclose(S[at_g], 36 / N_ELECTRONS, rtol=1e-5)
        np.testing.assert_allclose(connected, 0.0, atol=1e-4)     # a single fixed configuration
        self.assertEqual(sums["patterns"][code], 1)


if __name__ == "__main__":
    absltest.main()
