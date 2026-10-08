"""
Stage 7 checks.  Run:  python test_stage07.py

  1. the energy does not depend on the splitting parameter eta, and the
     default truncation is already converged.  eta is pure bookkeeping, so
     this one test exercises every formula in ewald.py at once: get any
     erfc argument, any factor of 2 or of 2 pi / A, the background or the
     self-energy term wrong, and the eta dependence no longer cancels.
  2. the Madelung constant reproduces published 2D Wigner-crystal values,
     for the triangular lattice we use AND for a square one.  eta cancels
     out of a wrong overall convention, so this is the external anchor.
  3. our energy agrees with the authors' make_2DCoulomb_potential, which is
     the implementation the paper's numbers came from.
  4. moving one electron by a supercell vector leaves the energy unchanged.
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
from slaternet import supercell_vectors  # noqa: E402
from ewald import make_ewald_setup, coulomb_energy  # noqa: E402

LATTICE = supercell_vectors(9)                 # triangular, 9 moire cells

# Energy per particle of a 2D lattice of unit charges in a neutralising
# background, in units of 1/r_s with r_s = 1/sqrt(pi n).  Bonsall and
# Maradudin, Phys. Rev. B 15, 1959 (1977) give -1.106103 for the triangular
# lattice; the extra digits (and the square-lattice value) are from the
# high-precision recomputation in arXiv:2103.01622, Table "eta_0".
WIGNER_CRYSTAL_ENERGY = {"triangular": -1.106102587, "square": -1.100244420}


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

    def test_madelung_constant_matches_the_published_wigner_crystal_values(self):
        """xi_M / 2 is the energy per particle of a lattice of unit charges.

        One charge per cell, so the density is n = 1 / area and
        r_s = 1 / sqrt(pi n).  Checked on a square lattice as well as on the
        triangular supercell, so a result that is only right for one
        geometry cannot pass.
        """
        lattices = {"triangular": LATTICE, "square": np.eye(2) * 2.3}
        for name, lattice in lattices.items():
            setup = make_ewald_setup(lattice, **self.converged)
            r_s = np.sqrt(setup.area / np.pi)
            np.testing.assert_allclose(0.5 * setup.madelung * r_s,
                                       WIGNER_CRYSTAL_ENERGY[name], rtol=1e-9)

    def test_agrees_with_the_authors_implementation(self):
        from periodicwave.pbc.hamiltonian import make_2DCoulomb_potential
        theirs = make_2DCoulomb_potential(jnp.asarray(LATTICE), None, None,
                                          truncation_limit=5, interaction_energy_scale=1.0)
        setup = make_ewald_setup(LATTICE)
        for seed in range(3):
            positions = six_electrons(seed)
            separations = positions[:, None, :] - positions[None, :, :]
            np.testing.assert_allclose(float(coulomb_energy(setup, positions)),
                                       float(theirs(None, separations)), rtol=1e-12)

    def test_moving_electrons_by_supercell_vectors_changes_nothing(self):
        """Each electron is displaced by its own, large, lattice vector.

        Not a formality: the Metropolis walkers are never folded back into the
        cell, and measured in Stage 7 they drift more than 5 supercells away
        within 3000 steps -- a production run is 30 times longer.  The shifts
        here (up to 9 cells) are deliberately bigger than the 5 shells of
        images that pair_potential sums over, so this fails unless
        wrap_into_supercell really does recentre every separation.
        """
        setup = make_ewald_setup(LATTICE)
        integers = np.array([[7, -9], [-6, 8], [9, 9], [-8, -7], [5, -6], [0, 0]])
        moved = self.positions + jnp.asarray(integers @ LATTICE.T)
        np.testing.assert_allclose(float(coulomb_energy(setup, moved)),
                                   float(coulomb_energy(setup, self.positions)), rtol=1e-12)


if __name__ == "__main__":
    absltest.main()
