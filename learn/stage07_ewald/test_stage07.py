"""
Stage 7 checks.  Run:  python test_stage07.py

  1. The energy does not depend on the splitting parameter eta.  This checks
     every term that contains eta: the erfc arguments, the background and the
     self-term of the Madelung constant.
  2. The authors' default truncation (5 shells of images) is converged.
  3. Perfect Wigner crystals have their published energies.  This is the one
     check that depends neither on eta nor on the authors' code, so it guards
     the pieces test 1 cannot see: the two factors of 1/2 in Eq. (A11), and
     how pair terms and self-image terms add up.
  4. Agreement with the authors at the paper's interaction strength: the
     value of U, and the energy from make_2DCoulomb_potential, the code
     behind the paper's numbers.
  5. Moving electrons by supercell vectors, farther than the window of images
     the sums use, changes nothing.
"""
import sys
from pathlib import Path

from absl.testing import absltest

import jax
jax.config.update("jax_enable_x64", True)     # before any JAX operation
import numpy as np
import jax.numpy as jnp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "stage03_slater"))
from slaternet import supercell_vectors, MOIRE_LATTICE  # noqa: E402
from ewald import make_ewald_setup, coulomb_energy, moire_interaction_strength  # noqa: E402

LATTICE = supercell_vectors(9)                 # triangular, 9 moire cells

# Classical energy per particle of a 2D lattice of unit charges on a
# neutralising background, in units of 1/r_s, r_s = sqrt(area per charge / pi).
# Bonsall and Maradudin, Phys. Rev. B 15, 1959 (1977): -1.106103 (triangular)
# and -1.100244 (square).  The extra digits are from the recomputation in
# arXiv:2103.01622 (its table of eta_0 for 2D lattices).
WIGNER_CRYSTAL = {"triangular": -1.106102587, "square": -1.100244420}


def six_electrons(seed):
    return jnp.asarray(np.random.default_rng(seed).uniform(0, 1, (6, 2)) @ LATTICE.T)


class EwaldSumTest(absltest.TestCase):

    def setUp(self):
        self.positions = six_electrons(0)
        # Enough images that every eta below is converged; the default is 5.
        self.converged = dict(real_space_shells=16, reciprocal_shells=16)

    def test_energy_does_not_depend_on_the_splitting_parameter(self):
        energies = [
            float(coulomb_energy(make_ewald_setup(LATTICE, gamma_factor=gamma, **self.converged),
                                 self.positions))
            for gamma in (1.6, 2.0, 2.8, 3.6, 5.0)
        ]
        np.testing.assert_allclose(energies, energies[0], rtol=1e-11)

    def test_the_default_truncation_is_converged(self):
        """The default is 5 shells; production runs must not need more."""
        default = float(coulomb_energy(make_ewald_setup(LATTICE), self.positions))
        converged = float(coulomb_energy(make_ewald_setup(LATTICE, **self.converged),
                                         self.positions))
        np.testing.assert_allclose(default, converged, rtol=1e-11)

    def test_perfect_wigner_crystals_have_the_published_energy(self):
        """Three crystals, each putting different weight on pairs and self-images.

        9 electrons on all 9 moire sites: a triangular crystal of spacing 1.
        3 electrons on one sqrt3 x sqrt3 sublattice: the same shape, spacing
            sqrt3, at a third of the density.
        4 electrons on a 2 x 2 grid in a square cell: a square crystal.
        """
        a1, a2 = MOIRE_LATTICE[:, 0], MOIRE_LATTICE[:, 1]
        grid = [(n, m) for n in range(3) for m in range(3)]
        all_sites = np.array([n * a1 + m * a2 for n, m in grid])
        sublattice = np.array([n * a1 + m * a2 for n, m in grid if (n + m) % 3 == 0])
        square_cell = 4.6 * np.eye(2)
        square_grid = 2.3 * np.array([[0, 0], [1, 0], [0, 1], [1, 1]])

        crystals = [(LATTICE, all_sites, "triangular"),
                    (LATTICE, sublattice, "triangular"),
                    (square_cell, square_grid, "square")]
        for lattice, positions, shape in crystals:
            setup = make_ewald_setup(lattice, **self.converged)
            n_electrons = len(positions)
            r_s = np.sqrt(setup.area / n_electrons / np.pi)
            energy = float(coulomb_energy(setup, jnp.asarray(positions)))
            np.testing.assert_allclose(energy / n_electrons * r_s, WIGNER_CRYSTAL[shape],
                                       rtol=1e-9)

    def test_agrees_with_the_authors_at_the_papers_interaction_strength(self):
        """U itself, then the energy, against the authors' code.

        10.623473362 and 5.311736681 are what the authors' convert_moire_scales()
        returns for eps = 5 and eps = 10 (that function cannot be imported on
        its own: its config file sets up a whole run when imported).  The
        energy is compared at U rather than at 1, so U has to multiply both
        terms of Eq. (A11).
        """
        np.testing.assert_allclose(moire_interaction_strength(dielectric_constant=5.0),
                                   10.623473362, rtol=1e-9)
        np.testing.assert_allclose(moire_interaction_strength(dielectric_constant=10.0),
                                   5.311736681, rtol=1e-9)

        from periodicwave.pbc.hamiltonian import make_2DCoulomb_potential
        U = moire_interaction_strength(dielectric_constant=5.0)
        theirs = make_2DCoulomb_potential(jnp.asarray(LATTICE), None, None,
                                          truncation_limit=5, interaction_energy_scale=U)
        setup = make_ewald_setup(LATTICE)
        for seed in range(3):
            positions = six_electrons(seed)
            separations = positions[:, None, :] - positions[None, :, :]
            np.testing.assert_allclose(float(coulomb_energy(setup, positions, strength=U)),
                                       float(theirs(None, separations)), rtol=1e-12)

    def test_moving_electrons_by_supercell_vectors_changes_nothing(self):
        """Each electron is displaced by its own, large, lattice vector.

        Not a formality: the Metropolis walkers are never folded back into the
        cell, and they drift more than 5 supercells away within 3000 steps; a
        training run is 30 times longer.  The shifts here (up to 9 cells) are
        larger than the 5 shells of images pair_potential sums over, so this
        fails unless wrap_into_supercell really recentres every separation.
        """
        setup = make_ewald_setup(LATTICE)
        integers = np.array([[7, -9], [-6, 8], [9, 9], [-8, -7], [5, -6], [0, 0]])
        moved = self.positions + jnp.asarray(integers @ LATTICE.T)
        np.testing.assert_allclose(float(coulomb_energy(setup, moved)),
                                   float(coulomb_energy(setup, self.positions)), rtol=1e-12)


if __name__ == "__main__":
    absltest.main()
