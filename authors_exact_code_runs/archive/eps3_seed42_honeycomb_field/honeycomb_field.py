"""
The honeycomb field for the first 20,000 iterations of the GPU 0 run (eps = 3, seed 42).

Their training script chooses its Hamiltonian by name: cfg.system.make_local_energy_fn is the
dotted name of a function, and their train.py imports it (train.py lines 522-532).  The script copy
run_2deg_triangular_potential_seed42_field.py names THIS module's local_energy.  It builds THEIR
local energy, unchanged (periodicwave.pbc.hamiltonian.local_energy: kinetic + moire + Coulomb), and
adds one term:

    E_field(r_1 .. r_18) = lambda * sum_i sum_{j=1,2,3} cos(kappa_j . r_i)

kappa_j: the three corners of the moire Brillouin zone at 0, 120 and 240 degrees, |kappa| = 4 pi / 3
(a_M = 1).  On the 27 moire sites (lattice points R, the minima of the moire potential) the sum over
j is 3 on one sublattice (the one holding the origin; sublattice 0 of
learn/.../checkpoint_structure.py) and -3/2 on the other two.  So the field raises the sites of
sublattice 0 by 4.5 lambda against the others: a honeycomb with sublattice 0 empty gains
18 x 1.5 lambda = 27 lambda, an even filling gains nothing.  kappa_j is one of the 27 momenta of the
supercell, so the field is periodic in it, as the rest of the Hamiltonian.

lambda = 0.5 code units = 1.69 meV: sublattice 0 raised by 7.6 meV; a full honeycomb gains
2.5 meV per electron from the field, far more than the 0.16 meV per electron by which the first
author's plot puts the Wigner crystal below the liquid near eps = 3.3.

The walkers are sampled from |Psi|^2 alone, so the field acts through the local energy: the energy
the network is trained to lower, and the energy reported in train_stats.csv while it is on.
"""
import jax.numpy as jnp
import numpy as np
from periodicwave.pbc import hamiltonian

KAPPAS = (4 * np.pi / 3) * np.array([[np.cos(a), np.sin(a)] for a in (0.0, 2 * np.pi / 3, 4 * np.pi / 3)])


def field_energy(positions, strength):
    """lambda * sum over electrons and the three kappas of cos(kappa . r); positions (2N,) or (N, 2)."""
    r = jnp.reshape(positions, (-1, 2))
    return strength * jnp.sum(jnp.cos(r @ jnp.asarray(KAPPAS, dtype=r.dtype).T))


def local_energy(f, charges, nspins, honeycomb_field_strength=0.0, **their_arguments):
    """Their local energy (same arguments) plus the field.  Returns e_l(params, key, data) like theirs."""
    theirs = hamiltonian.local_energy(f, charges, nspins, **their_arguments)

    def e_l(params, key, data):
        energy, extra = theirs(params, key, data)
        return energy + field_energy(data.positions, honeycomb_field_strength), extra
    return e_l
