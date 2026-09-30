"""
Stage 6 checks.  Run:  python test_stage06.py

  1. the exact benchmark reproduces a case we know by hand
  2. the energy gradient equals the exact gradient -- and dropping Im E_loc
     does not, unless the wave function is real
  3. the natural-gradient step solves (S + damping) x = gradient, respects
     the size limit, and stays finite and downhill when S has rounding errors
"""
import sys
from pathlib import Path

from absl.testing import absltest

import jax
jax.config.update("jax_enable_x64", True)     # before any JAX operation
import numpy as np
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "stage03_slater"))
sys.path.insert(0, str(HERE.parent / "stage04_mcmc"))
from slaternet import Architecture, LogPsi, init_parameters, supercell_vectors  # noqa: E402
from local_energy import moire_units, local_energy, plane_wave_k_vectors  # noqa: E402
from exact_noninteracting import (exact_ground_state_energy, plane_wave_basis,  # noqa: E402
                                  single_particle_hamiltonian)
from gradient import natural_gradient_step  # noqa: E402
from check_phase_gradient import grid_points, exact_energy, estimators  # noqa: E402


class ExactBenchmarkTest(absltest.TestCase):

    def test_without_potential_gives_the_plane_wave_energy(self):
        """With V0 = 0 the answer is the Stage 4 plane-wave determinant: sum of |k|^2 / 2."""
        lattice = supercell_vectors(9)
        e0, _ = exact_ground_state_energy(lattice, 6, strength=0.0, phi=np.pi / 4)
        k = plane_wave_k_vectors(lattice, 6)
        np.testing.assert_allclose(e0, 0.5 * np.sum(k**2), rtol=1e-12)

    def test_lowest_state_agrees_with_the_stage4_hamiltonian(self):
        """The plane-wave solver and Stage 4's real-space local energy must describe the same h.

        Build the lowest eigenstate psi(r) = sum_q c_q exp(i q.r) from the solver,
        and evaluate Stage 4's E_loc = (h psi)/psi at random points.  For an exact
        eigenstate E_loc equals the eigenvalue everywhere.  A sign slip in the
        potential matrix elements (e.g. e^{+i phi} vs e^{-i phi}) breaks this.
        """
        # Cutoff 50, not 30: the eigenvalue converges fast, but E_loc at a point
        # is a stricter test.  Measured max |E_loc - eigenvalue|: 8e-4 at cutoff
        # 30, 2e-6 at 40, 1e-9 at 50.  A wrong sign of phi gives 29 at any cutoff.
        _, strength = moire_units()
        phi = np.pi / 4
        lattice = supercell_vectors(9)
        H = single_particle_hamiltonian(lattice, strength, phi, cutoff=50.0)
        _, q = plane_wave_basis(lattice, cutoff=50.0)
        eigenvalues, eigenvectors = np.linalg.eigh(H)
        coefficients = jnp.asarray(eigenvectors[:, 0])
        q = jnp.asarray(q)

        def orbital(R):                                   # R: (1, 2), one electron
            psi = jnp.sum(coefficients * jnp.exp(1j * (q @ R[0])))
            return LogPsi(log_abs=jnp.log(jnp.abs(psi)), phase=jnp.angle(psi))

        rng = np.random.default_rng(0)
        for _ in range(5):
            R = jnp.asarray(rng.uniform(0, 1, (1, 2)) @ lattice.T)
            e_loc = complex(local_energy(orbital, R, strength, phi))
            np.testing.assert_allclose(e_loc.real, eigenvalues[0], rtol=1e-7)
            np.testing.assert_allclose(e_loc.imag, 0.0, atol=1e-7)


class GradientTest(absltest.TestCase):

    def setUp(self):
        arch = Architecture(n_electrons=1, n_layers=2, width=32, init_scale=0.3)
        self.complex_params = init_parameters(jax.random.key(0), arch)
        self.real_params = dict(self.complex_params,
                                proj=self.complex_params["proj"].at[:, :, 1, :].set(0.0))
        self.points = grid_points(40)

    def exact_gradient(self, params):
        return ravel_pytree(jax.grad(exact_energy)(params, self.points))[0]

    def test_full_estimator_equals_the_exact_gradient(self):
        exact = self.exact_gradient(self.complex_params)
        results, _ = estimators(self.complex_params, self.points)
        np.testing.assert_allclose(results["full (gradient.py)"], exact, rtol=1e-6, atol=1e-8)

    def test_dropping_im_e_loc_is_wrong_for_a_complex_wave_function(self):
        exact = self.exact_gradient(self.complex_params)
        results, _ = estimators(self.complex_params, self.points)
        error = np.linalg.norm(results["real only (gradient.py)"] - exact) / np.linalg.norm(exact)
        self.assertGreater(error, 0.1)

    def test_dropping_im_e_loc_is_harmless_for_a_real_wave_function(self):
        exact = self.exact_gradient(self.real_params)
        results, _ = estimators(self.real_params, self.points)
        np.testing.assert_allclose(results["real only (gradient.py)"], exact, rtol=1e-6, atol=1e-8)


class NaturalGradientStepTest(absltest.TestCase):

    def setUp(self):
        rng = np.random.default_rng(0)
        A = rng.normal(size=(20, 40))
        self.S = jnp.asarray(A @ A.T / 40)                # positive definite, full rank
        self.gradient = jnp.asarray(rng.normal(size=20))

    def test_small_step_solves_the_damped_system(self):
        step = natural_gradient_step(self.gradient, self.S, learning_rate=1e-4,
                                     damping=1e-2, max_step_norm=1e6)
        np.testing.assert_allclose((self.S + 1e-2 * jnp.eye(20)) @ step, -1e-4 * self.gradient,
                                   rtol=1e-8, atol=1e-12)

    def test_large_step_is_capped(self):
        step = natural_gradient_step(self.gradient, self.S, learning_rate=10.0,
                                     damping=1e-2, max_step_norm=1e-3)
        np.testing.assert_allclose(step @ self.S @ step, 1e-3, rtol=1e-8)

    def test_rounding_errors_in_S_cannot_turn_the_step_uphill(self):
        """Give S a negative eigenvalue larger than the damping, as float32 did.

        The gradient points mostly along that eigenvector: the case where a
        naive (S + damping)^-1 would divide by a negative number and go uphill.
        """
        eigenvalues, eigenvectors = np.linalg.eigh(np.asarray(self.S))
        eigenvalues[0] = -0.9
        broken_S = jnp.asarray(eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T)
        gradient = jnp.asarray(eigenvectors[:, 0]) + 0.01 * self.gradient / jnp.linalg.norm(self.gradient)

        step = natural_gradient_step(gradient, broken_S, learning_rate=0.1,
                                     damping=1e-3, max_step_norm=1e-2)
        self.assertTrue(bool(jnp.all(jnp.isfinite(step))))
        self.assertLess(float(step @ gradient), 0.0)           # downhill


if __name__ == "__main__":
    absltest.main()
