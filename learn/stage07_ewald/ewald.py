"""
Stage 7 -- the Coulomb repulsion between electrons, under periodic boundaries.

Paper:   Appendix A, Eqs. (A2)-(A12)
Authors: pbc/hamiltonian.py, make_2DCoulomb_potential

Stages 4-6 ran on H = kinetic energy + moire potential.  This stage adds the
last term of Eq. (1), the repulsion between electrons.  That term is the
whole point of the paper: without it the electrons are independent and a
single Slater determinant is exact (Stage 6 used precisely that fact to get
an exact answer to aim at).


WHY IT IS NOT SIMPLY  sum 1/r  OVER PAIRS
.........................................
Our electrons live in one supercell of a lattice that tiles the plane, so
electron a repels not only electron b but every periodic image of b -- and
every image of itself.  That is Eq. (A2):

    H_ee = 1/2 sum_{a != b} sum_L  1 / |r_a - r_b + L|          (other electrons)
         + 1/2 sum_a sum_{L != 0}  1 / |L|                      (own images)

where the second line, per electron, is the Madelung constant xi_M, and

L runs over all supercell lattice vectors.  Both lines diverge.  In 2D
a ring of radius R and width dR holds 2 pi R dR / A lattice points, and each
contributes about 1/R, so the partial sums grow like 2 pi R / A -- linearly,
without bound.

The divergence is not physical: a real sample is neutral.  Putting the
electrons on a uniform background of opposite charge removes the q = 0
Fourier component of the interaction, and with it the divergence.  In the
paper this appears as a constant subtracted from every pair (Eq. A8):

    phi_background = (2 pi / A) * 2 eta / sqrt(pi).


EWALD'S TRICK
.............
Even once it is finite, a direct real-space sum converges far too slowly to
use.  Ewald splits the interaction into two halves that each converge fast,
but in different places.  Using  1 = erfc(x) + erf(x),

    1/|r|  =  erfc(|r| / 2 eta) / |r|     short ranged: killed by a Gaussian
            + erf (|r| / 2 eta) / |r|     long ranged, but perfectly smooth

The short-ranged half is summed over images in real space and is negligible
past a few eta (Eq. A6).  The smooth half is summed in reciprocal space,
where smooth means that only small |G| matter (Eq. A10).

eta decides where the split falls: small eta moves work into the reciprocal
sum, large eta into the real-space sum.  The physics cannot depend on it,
and test_stage07.py checks exactly that -- one test that exercises every
formula in this file at once.

The authors write the same split with  gamma = 1 / (2 eta)^2  and choose
gamma_factor = 2.8 (their comment cites the CASINO QMC manual), which is
roughly the value that minimises the work for a given accuracy.
"""
import dataclasses

import numpy as np
import scipy.special
import jax
import jax.numpy as jnp

# The authors' default (pbc/hamiltonian.py).  gamma = (gamma_factor / sqrt(area))^2.
DEFAULT_GAMMA_FACTOR = 2.8


# ---------------------------------------------------------------------------
#  Unit conversion: how strong is the repulsion, in code units?
# ---------------------------------------------------------------------------

def moire_interaction_strength(effective_mass=0.35, lattice_constant_nm=8.031,
                               dielectric_constant=5.0):
    """U in  H_ee = (U/2) sum_{i != j} 1/|r_i - r_j|,  lengths in units of a_M.

    Same formula as convert_moire_scales() in the authors'
    run_2deg_triangular_potential.py:  U = (a_M / a_0) * (1/eps) * (m* / m_e).
    It is the ratio of the Coulomb scale e^2/(eps a_M) to the kinetic scale
    hbar^2/(m* a_M^2), so a heavier electron or a weaker screen means a more
    strongly interacting problem.  Paper values: 10.623473 at eps = 5,
    5.311737 at eps = 10.
    """
    bohr_radius_nm = 5.29177210544e-2
    return (lattice_constant_nm / bohr_radius_nm) * effective_mass / dielectric_constant


# ---------------------------------------------------------------------------
#  Setting up: everything that depends only on the supercell
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class EwaldSetup:
    """Fixed geometry for one supercell.  Build it once, then reuse it."""
    lattice: np.ndarray          # (2, 2), columns are L1 and L2
    inverse_lattice: np.ndarray  # (2, 2)
    area: float
    eta: float                   # the splitting parameter of Eq. (A5)
    images: jnp.ndarray          # (n_images, 2) lattice vectors L, zero included
    g_vectors: jnp.ndarray       # (n_g, 2) reciprocal vectors G, zero excluded
    g_weights: jnp.ndarray       # (n_g,) erfc(eta |G|) / |G|, precomputed
    background: float            # phi_background, Eq. (A8)
    madelung: float              # xi_M, Eq. (A12), background already subtracted


def _lattice_points(basis, n_shells):
    """All basis @ (n, m) with |n|, |m| <= n_shells.  The zero vector comes first."""
    ordinals = np.array([(n, m)
                         for n in range(-n_shells, n_shells + 1)
                         for m in range(-n_shells, n_shells + 1)])
    ordinals = ordinals[np.argsort(np.abs(ordinals).sum(axis=1), kind="stable")]
    points = ordinals @ basis.T                    # points[i] = basis @ ordinals[i]
    assert np.allclose(points[0], 0.0)
    return points


def make_ewald_setup(lattice, gamma_factor=DEFAULT_GAMMA_FACTOR,
                     real_space_shells=5, reciprocal_shells=5):
    """Precompute the image lists and the two constants of Eqs. (A8) and (A12).

    lattice: (2, 2) with the supercell vectors L1, L2 as COLUMNS.
    The shell counts match the authors' truncation_limit = 5, i.e. 11 x 11
    images in each sum.  They need to be larger when eta is far from its
    default, which is why the eta test raises them.
    """
    lattice = np.asarray(lattice, dtype=float)
    area = float(abs(np.linalg.det(lattice)))
    eta = np.sqrt(area) / (2.0 * gamma_factor)

    images = _lattice_points(lattice, real_space_shells)
    reciprocal = 2.0 * np.pi * np.linalg.inv(lattice)       # rows are G1, G2
    g_vectors = _lattice_points(reciprocal.T, reciprocal_shells)[1:]   # drop G = 0
    g_norms = np.linalg.norm(g_vectors, axis=1)

    # Eq. (A8): the q -> 0 piece of the short-range potential, removed from
    # every pair by the neutralising background.
    background = (2.0 * np.pi / area) * (2.0 * eta / np.sqrt(np.pi))

    # Eq. (A12): the energy of one electron in the field of its OWN images.
    # Same two sums as a pair, but with the L = 0 and G = 0 terms handled
    # differently: the real-space sum skips L = 0 (an electron does not feel
    # itself), and the long-range sum must give back the self-energy of the
    # smeared charge it introduced, xi^L_0 = 1 / (eta sqrt(pi)).
    distances = np.linalg.norm(images[1:], axis=1)
    self_image_long_range = 1.0 / (eta * np.sqrt(np.pi))
    madelung = float(
        np.sum(scipy.special.erfc(distances / (2.0 * eta)) / distances)
        + (2.0 * np.pi / area) * np.sum(scipy.special.erfc(eta * g_norms) / g_norms)
        - self_image_long_range
        - background)

    return EwaldSetup(lattice=lattice,
                      inverse_lattice=np.linalg.inv(lattice),
                      area=area,
                      eta=float(eta),
                      images=jnp.asarray(images),
                      g_vectors=jnp.asarray(g_vectors),
                      g_weights=jnp.asarray(scipy.special.erfc(eta * g_norms) / g_norms),
                      background=float(background),
                      madelung=madelung)


# ---------------------------------------------------------------------------
#  The energy of one configuration
# ---------------------------------------------------------------------------

def wrap_into_supercell(setup, vectors):
    """Shift each vector by a lattice vector into the cell centred on the origin.

    The image sums are truncated at a few shells around the separation, so
    the separation itself has to be small; otherwise the window of images
    sits off-centre and the sums lose accuracy.  Shifting by a lattice vector
    changes nothing physically, because the sum over images is shifted too.
    """
    fractional = vectors @ setup.inverse_lattice.T
    fractional = (fractional + 0.5) % 1.0 - 0.5
    return fractional @ setup.lattice.T


def pair_potential(setup, separation):
    """phi(r): potential at r from a unit charge at the origin and all its images.

    Eq. (A6) + Eq. (A10) - Eq. (A8): short range + long range - background.
    Infinite at separation zero (the charge's own position); coulomb_energy
    never uses that value.
    """
    distances = jnp.linalg.norm(separation - setup.images, axis=-1)
    short_range = jnp.sum(jax.scipy.special.erfc(distances / (2.0 * setup.eta)) / distances)

    # exp(i G.r) summed over a G set that contains both +G and -G is just
    # cos(G.r), so no complex arithmetic is needed.
    long_range = (2.0 * jnp.pi / setup.area) * jnp.sum(
        jnp.cos(setup.g_vectors @ separation) * setup.g_weights)

    return short_range + long_range - setup.background


def coulomb_energy(setup, positions, strength=1.0):
    """H_ee for one configuration of N electrons: Eq. (A11), times `strength`.

    positions: (N, 2).  With strength = 1 the pair interaction is 1/r; pass
    moire_interaction_strength() to get the paper's U/2 sum_{i != j} 1/r.
    """
    n_electrons = positions.shape[0]
    separations = wrap_into_supercell(setup, positions[:, None, :] - positions[None, :, :])

    pair = jax.vmap(jax.vmap(pair_potential, in_axes=(None, 0)), in_axes=(None, 0))(
        setup, separations)
    pair = pair.at[jnp.diag_indices(n_electrons)].set(0.0)   # drop a = b: handled by xi_M

    # First term of Eq. (A11): each distinct pair once, in the potential of
    # the other electron and all its images.  Second term: each electron in
    # the potential of its own images.
    return strength * (0.5 * jnp.sum(pair) + 0.5 * n_electrons * setup.madelung)


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "stage03_slater"))
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "stage04_mcmc"))
    jax.config.update("jax_enable_x64", True)
    from slaternet import supercell_vectors
    from local_energy import moire_units

    energy_unit_meV, _ = moire_units()
    lattice = supercell_vectors(9)
    setup = make_ewald_setup(lattice)

    print(f"supercell area        {setup.area:.6f} a_M^2   (9 moire cells)")
    print(f"splitting parameter   eta = {setup.eta:.6f} a_M")
    print(f"background constant   {setup.background:.6f}")
    print(f"Madelung constant     xi_M = {setup.madelung:.6f} / a_M "
          f"= {setup.madelung * energy_unit_meV:+.4f} meV")

    # Six electrons on a triangular sublattice of the 9-site cell: the
    # generalised Wigner crystal arrangement at filling 2/3.
    rng = np.random.default_rng(0)
    positions = jnp.asarray(rng.uniform(0, 1, (6, 2)) @ lattice.T)
    print("\nrandom configuration of 6 electrons:")
    for dielectric in (5.0, 10.0):
        U = moire_interaction_strength(dielectric_constant=dielectric)
        energy = float(coulomb_energy(setup, positions, strength=U))
        print(f"  eps = {dielectric:4.1f}   U = {U:9.6f}   "
              f"H_ee = {energy:+.6f} code units = {energy * energy_unit_meV:+.3f} meV")
