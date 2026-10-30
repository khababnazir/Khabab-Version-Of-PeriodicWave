"""
Stage 4 checks.  Run:  python test_stage04.py

Three checks, one per idea in this stage:
  1. the kinetic energy of an exact eigenstate comes out exact
  2. our moire potential equals the authors' and has the lattice's period
  3. the Metropolis sampler reproduces a distribution whose answer we know

Written with absltest, the test framework FermiNet and PeriodicWave's
upstream use.  np.testing.assert_allclose fails if a value is NaN, so a
broken result cannot slip through as a pass.
"""
import sys
from pathlib import Path

from absl.testing import absltest

import jax
jax.config.update("jax_enable_x64", True)     # before any JAX operation
import numpy as np
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "stage03_slater"))
from slaternet import MOIRE_LATTICE as MOIRE_LATTICE_VECTORS, supercell_vectors  # noqa: E402
from slaternet import (Architecture, init_parameters, orbital_matrices, reciprocal_vectors,  # noqa: E402
                       sum_of_determinants, sum_of_determinants_slogdet)
from local_energy import (moire_units, moire_potential, kinetic_energy,  # noqa: E402
                          kinetic_energy_folx, plane_wave_k_vectors, plane_wave_log_psi)
from metropolis import run_metropolis  # noqa: E402
from periodicwave.pbc.hamiltonian import make_triangular_potential  # noqa: E402


class KineticEnergyTest(absltest.TestCase):

    def test_plane_waves_are_an_exact_eigenstate(self):
        """A plane-wave determinant must give E_kin = sum |k|^2 / 2 at every configuration.

        This checks every term of the kinetic-energy formula at once: the
        wave function has both a varying magnitude and a varying phase, so a
        wrong sign or a missing term would make the answer wrong or make it
        change from configuration to configuration.
        """
        lattice = supercell_vectors(9)
        k = plane_wave_k_vectors(lattice, 6)
        exact = 0.5 * np.sum(k**2)
        rng = np.random.default_rng(0)

        for _ in range(5):
            positions = jnp.asarray(rng.uniform(0, 1, (6, 2)) @ lattice.T)
            e_kin = complex(kinetic_energy(lambda R: plane_wave_log_psi(R, k), positions))
            np.testing.assert_allclose(e_kin.real, exact, rtol=1e-10)
            np.testing.assert_allclose(e_kin.imag, 0.0, atol=1e-10)

    def test_folx_gives_the_same_kinetic_energy(self):
        """Stage 8: kinetic_energy_folx must equal kinetic_energy, real and
        imaginary part: on the exact eigenstate above, and on a 4-determinant
        SlaterNet whose magnitude and phase both vary, with its determinants
        added either way (det or slogdet)."""
        lattice = supercell_vectors(9)
        G = reciprocal_vectors(lattice)
        k = plane_wave_k_vectors(lattice, 6)
        params = init_parameters(jax.random.key(3), Architecture(n_electrons=6, n_determinants=4))
        wave_functions = {
            "plane waves": lambda R: plane_wave_log_psi(R, k),
            "SlaterNet, det": lambda R: sum_of_determinants(orbital_matrices(params, R, G)),
            "SlaterNet, slogdet": lambda R: sum_of_determinants_slogdet(orbital_matrices(params, R, G)),
        }
        rng = np.random.default_rng(1)
        for name, log_psi_fn in wave_functions.items():
            for _ in range(3):
                positions = jnp.asarray(rng.uniform(0, 1, (6, 2)) @ lattice.T)
                hessian = complex(kinetic_energy(log_psi_fn, positions))
                folx = complex(kinetic_energy_folx(log_psi_fn, positions))
                with self.subTest(name):
                    np.testing.assert_allclose(folx.real, hessian.real, rtol=0, atol=1e-10 * abs(hessian))
                    np.testing.assert_allclose(folx.imag, hessian.imag, rtol=0, atol=1e-10 * abs(hessian))


class MoirePotentialTest(absltest.TestCase):

    def test_matches_the_authors_potential(self):
        _, strength = moire_units()
        phi = np.pi / 4
        authors = make_triangular_potential(MOIRE_LATTICE_VECTORS, None, None, strength, phi)

        positions = jnp.asarray(np.random.default_rng(1).uniform(-2, 2, (6, 2)))
        np.testing.assert_allclose(moire_potential(positions, strength, phi),
                                   authors(positions), rtol=1e-12)

    def test_has_the_period_of_the_moire_lattice(self):
        _, strength = moire_units()
        phi = np.pi / 4
        positions = jnp.asarray(np.random.default_rng(2).uniform(-2, 2, (6, 2)))
        reference = moire_potential(positions, strength, phi)

        for a in (MOIRE_LATTICE_VECTORS[:, 0], MOIRE_LATTICE_VECTORS[:, 1]):
            shifted = moire_potential(positions + a, strength, phi)
            np.testing.assert_allclose(shifted, reference, rtol=1e-12)


class MetropolisTest(absltest.TestCase):

    def test_samples_a_known_distribution(self):
        """Sample |Psi|^2 = a Gaussian with variance sigma^2 and check its moments.

        With log|Psi| = -|x|^2 / (4 sigma^2), |Psi|^2 is proportional to
        exp(-|x|^2 / (2 sigma^2)): each coordinate has mean 0, variance sigma^2.
        Burn in with step-width tuning, then sample with the width frozen --
        the same recipe estimate_energy.py uses.  The walkers are independent
        chains, so the final snapshot is a clean sample of 4096 walkers x 2
        coordinates.  Its variance has a statistical
        uncertainty of about sigma^2 * sqrt(2 / 8192), roughly 1.6%, so 6% is a
        safe tolerance.
        """
        sigma = 0.7

        def log_abs_psi_batch(walkers):                 # walkers: (n_walkers, 1, 2)
            return -jnp.sum(walkers**2, axis=(1, 2)) / (4 * sigma**2)

        walkers = jnp.zeros((4096, 1, 2))
        walkers, width, _ = run_metropolis(log_abs_psi_batch, walkers, jax.random.key(0),
                                           n_steps=200, step_width=0.1)
        walkers, frozen_width, rates = run_metropolis(log_abs_psi_batch, walkers, jax.random.key(1),
                                                      n_steps=100, step_width=width, adapt_every=None)
        self.assertEqual(frozen_width, width)       # adapt_every=None really keeps it fixed
        samples = np.asarray(walkers).reshape(-1)

        np.testing.assert_allclose(samples.mean(), 0.0, atol=0.03)
        np.testing.assert_allclose(samples.var(), sigma**2, rtol=0.06)
        self.assertBetween(np.mean(rates), 0.40, 0.65)   # the tuned width is reasonable


if __name__ == "__main__":
    absltest.main()
