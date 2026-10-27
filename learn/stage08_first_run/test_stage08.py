"""
Stage 8 check.  Run:  python test_stage08.py      (about a minute, on the CPU)

From this stage on, the paper's numbers come from the authors' code, not from
learn/.  Everything we built in Stages 3-7 is only useful as a cross-check if
both codes integrate the SAME Hamiltonian.  This test evaluates the authors'
local energy (kinetic + moire potential + Ewald Coulomb, at the paper's U for
eps = 5) and ours, on one wave function -- the SlaterNet trained in Stage 6 --
at configurations sampled from its |Psi|^2.

It checks both of the authors' kinetic-energy paths:
  folx     (what their moire config uses): must equal the REAL part of ours,
           with the imaginary part exactly zero -- it is computed and then
           discarded by jnp.real (pbc/hamiltonian.py);
  default  must equal ours completely, imaginary part included.

What this does not show.  The test is about the local ENERGY.  Whether the
authors' training would then use the imaginary part correctly in the
GRADIENT is a separate question about their loss.py.  Stage 6 reproduced
loss.py line by line and found it correct when clipping is on
(check_phase_gradient.py), but did not run their code.  That has to be tested
directly before a 'default' run is used to decide whether dropping Im E_loc
matters.  Nor does this test cover the optimiser, checkpoints, or convergence.
"""
import json
import os
import sys
from pathlib import Path

from absl.testing import absltest

os.environ["JAX_PLATFORMS"] = "cpu"          # before JAX is imported: a GPU may be training
import jax
jax.config.update("jax_enable_x64", True)     # before any JAX operation
import numpy as np
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

LEARN = Path(__file__).resolve().parent.parent
for stage in ("stage03_slater", "stage04_mcmc", "stage07_ewald"):
    sys.path.insert(0, str(LEARN / stage))
from slaternet import Architecture, init_parameters, log_psi, supercell_vectors, reciprocal_vectors  # noqa: E402
from local_energy import moire_units, local_energy  # noqa: E402
from ewald import make_ewald_setup, coulomb_energy, moire_interaction_strength  # noqa: E402

from periodicwave import networks  # noqa: E402
from periodicwave.pbc import hamiltonian as authors_hamiltonian, lattices  # noqa: E402

STAGE6_RUN = LEARN.parent / "runs" / "stage06" / "full"
PHI = np.pi / 4


class SameHamiltonianTest(absltest.TestCase):

    @classmethod
    def setUpClass(cls):
        if not (STAGE6_RUN / "params.npz").exists():
            raise absltest.SkipTest(f"needs the Stage 6 run in {STAGE6_RUN}")
        _, cls.V = moire_units()
        cls.U = moire_interaction_strength(dielectric_constant=5.0)

        cls.supercell, cls.moire_lattice = lattices._triangular_lattice_vecs_periodic_potential(1.0, 9)
        np.testing.assert_allclose(cls.supercell, supercell_vectors(9))     # same 9-site cell
        cls.G = reciprocal_vectors(cls.supercell)

        # The Stage 6 wave function, cast to float64 (training saved it in float32).
        architecture = Architecture(**json.loads((STAGE6_RUN / "summary.json").read_text())["architecture"])
        _, unflatten = ravel_pytree(init_parameters(jax.random.key(0), architecture))
        saved = np.load(STAGE6_RUN / "params.npz")
        cls.params = unflatten(jnp.asarray(saved["flat_params"], dtype=jnp.float64))
        cls.walkers = jnp.asarray(saved["walkers"][:16], dtype=jnp.float64)

        # Ours: Stage 4 (kinetic + moire) plus Stage 7 (Coulomb).
        setup = make_ewald_setup(cls.supercell)
        psi = lambda R: log_psi(cls.params, R, cls.G)                          # noqa: E731
        cls.ours = np.array([complex(local_energy(psi, R, cls.V, PHI)
                                     + coulomb_energy(setup, R, strength=cls.U))
                             for R in cls.walkers])

    def authors_local_energy(self, laplacian_method):
        """The authors' local energy, with our wave function in their calling convention."""
        def network(params, x, spins, atoms, charges):
            out = log_psi(params, jnp.reshape(x, (-1, 2)), self.G)
            return out.phase, out.log_abs

        e_l = authors_hamiltonian.local_energy(
            network, jnp.array([0.0]), (6, 0), use_scan=False, complex_output=True,
            lattice=jnp.asarray(self.supercell), potential_type="CoulombMoire",
            potential_kwargs={"moire_lattice_vectors": self.moire_lattice,
                              "moire_potential_strength": self.V,
                              "moire_potential_phi": PHI,
                              "interaction_energy_scale": self.U},
            kinetic_kwargs={"laplacian_method": laplacian_method})
        values = []
        for R in self.walkers:
            data = networks.FermiNetData(positions=jnp.reshape(R, (-1,)), spins=jnp.ones(6),
                                         atoms=jnp.zeros((1, 2)), charges=jnp.array([0.0]))
            values.append(complex(e_l(self.params, None, data)[0]))
        return np.array(values)

    def test_folx_path_is_the_real_part_of_our_local_energy(self):
        theirs = self.authors_local_energy("folx")
        np.testing.assert_allclose(theirs.real, self.ours.real, rtol=1e-11)
        np.testing.assert_array_equal(theirs.imag, 0.0)
        self.assertGreater(np.abs(self.ours.imag).max(), 1e-2)    # so something WAS dropped

    def test_default_path_equals_our_local_energy_including_the_imaginary_part(self):
        theirs = self.authors_local_energy("default")
        np.testing.assert_allclose(theirs.real, self.ours.real, rtol=1e-11)
        np.testing.assert_allclose(theirs.imag, self.ours.imag, rtol=1e-9, atol=1e-12)


if __name__ == "__main__":
    absltest.main()
