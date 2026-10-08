"""
Stage 7 -- the Coulomb repulsion between electrons, under periodic boundaries.

Paper:   Appendix A, Eqs. (A2)-(A12)
Authors: pbc/hamiltonian.py, make_2DCoulomb_potential

Stages 4-6 used H = kinetic energy + moire potential.  This stage adds the
last term of the paper's Eq. (1), the repulsion between electrons.  That term
is what makes the problem hard: without it the electrons are independent and
a single Slater determinant is the exact ground state (Stage 6 used exactly
that fact to get an exact answer to aim at).


WHY IT IS NOT SIMPLY  sum 1/r  OVER PAIRS
.........................................
Our electrons live in one supercell of a lattice of supercells that tiles the
plane.  So electron a repels electron b and every periodic image of b, and it
also repels its own images.  That is Eq. (A2):

    H_ee = 1/2 sum_{a != b} sum_L  1 / |r_a - r_b + L|     the other electrons
         + 1/2 sum_a  sum_{L != 0} 1 / |L|                 each electron's own images

L runs over all supercell lattice vectors.  The inner sum of the second line
is the Madelung constant xi_M.  Both lines diverge: in 2D a ring of radius R
and width dR holds 2 pi R dR / A lattice points, each contributing about 1/R,
so the partial sums grow like 2 pi R / A, without bound.

The divergence is not physical.  A real sample is neutral, and the paper puts
the electrons on a uniform background of opposite charge.  The background
cancels the q = 0 Fourier component of the interaction, which is precisely
the divergent part.  Below, that shows up twice: the reciprocal-space sum
leaves out G = 0, and the constant of Eq. (A8),

    phi_background = (2 pi / A) * 2 eta / sqrt(pi),

is subtracted from the potential of every charge.  A consequence worth
knowing: the potential of one charge, its images and the background averages
to zero over the cell.


EWALD'S TRICK
.............
Even with the background in place, a sum over images in real space converges
far too slowly to use.  Ewald splits the interaction into two pieces that each
converge fast, in different spaces.  With  1 = erfc(x) + erf(x),

    1/|r|  =  erfc(|r| / 2 eta) / |r|     short ranged: decays like a Gaussian
            + erf (|r| / 2 eta) / |r|     long ranged, but smooth everywhere

The short-ranged piece is summed over images in real space (Eq. A6); at a
distance of 10 eta it is already down to erfc(5) = 1.5e-12.  The smooth piece
is summed over reciprocal vectors G (Eq. A10), where each term carries
erfc(eta |G|), so only small |G| contribute.

eta only decides how the work is shared: a small eta makes the real-space sum
short and the reciprocal sum long, a large eta the opposite.  The total cannot
depend on it, and test_stage07.py checks that it does not.  That check catches
mistakes in every term that contains eta (the erfc arguments, the background,
the self-term of the Madelung constant), but not in the pieces without eta,
such as the factors of 1/2 or the interaction strength.  Those are checked
against perfect Wigner crystals, whose energies are known, and against the
authors' code.

The authors write the same split with  gamma = 1 / (2 eta)^2  and set
gamma_factor = 2.8 (their comment refers to the CASINO QMC manual).  With
their 5 shells of images the energy agrees with a 16-shell calculation to
better than 1e-11 (also tested).
"""
import dataclasses

import numpy as np
import scipy.special
import jax
import jax.numpy as jnp

# The authors' default (pbc/hamiltonian.py): gamma = (gamma_factor / sqrt(area))^2.
DEFAULT_GAMMA_FACTOR = 2.8


# ---------------------------------------------------------------------------
#  Unit conversion: how strong is the repulsion, in code units?
# ---------------------------------------------------------------------------

def moire_interaction_strength(effective_mass=0.35, lattice_constant_nm=8.031,
                               dielectric_constant=5.0):
    """U in  H_ee = (U/2) sum_{i != j} 1/|r_i - r_j|,  lengths in units of a_M.

    U is the Coulomb energy scale e^2 / (4 pi eps_0 eps a_M) divided by the
    kinetic energy scale hbar^2 / (m* a_M^2), which works out to

        U = (a_M / a_0) * (m* / m_e) / eps,        a_0 = Bohr radius.

    A heavier electron, a longer moire period or weaker screening all make
    the interaction stronger.  Same formula and same a_0 as the authors'
    convert_moire_scales() in run_2deg_triangular_potential.py, which gives
    U = 10.623473362 at eps = 5 and 5.311736681 at eps = 10.  (The paper
    itself quotes eps, not U.)
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
    g_vectors: jnp.ndarray       # (n_g, 2) reciprocal vectors G, zero left out
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

    lattice: (2, 2), the supercell vectors L1 and L2 as COLUMNS.
    The default shell counts match the authors' truncation_limit = 5: the
    11 x 11 lattice vectors with |n|, |m| <= 5 in each sum, with G = 0 left
    out of the reciprocal one.  Far from the default eta more shells are
    needed, which is why the eta test raises them.
    """
    lattice = np.asarray(lattice, dtype=float)
    area = float(abs(np.linalg.det(lattice)))
    eta = np.sqrt(area) / (2.0 * gamma_factor)

    images = _lattice_points(lattice, real_space_shells)
    reciprocal = 2.0 * np.pi * np.linalg.inv(lattice)                  # rows are G1, G2
    g_vectors = _lattice_points(reciprocal.T, reciprocal_shells)[1:]   # leave out G = 0
    g_norms = np.linalg.norm(g_vectors, axis=1)
    g_weights = scipy.special.erfc(eta * g_norms) / g_norms

    # Eq. (A8): the q -> 0 part of the short-range potential, cancelled by
    # the neutralising background.
    background = (2.0 * np.pi / area) * (2.0 * eta / np.sqrt(np.pi))

    # Eq. (A12): xi_M is the potential an electron feels at its own position
    # from everything except its own bare 1/r, i.e. from its images and the
    # background.  Two terms differ from a pair:
    #   real space:        L = 0 is the electron itself, so skip it;
    #   reciprocal space:  the sum also contains the smooth piece of the
    #                      electron's OWN 1/r, which at r = 0 equals
    #                      erf(r / 2 eta) / r  ->  1 / (eta sqrt(pi))
    #                      (the paper's xi^L_0), so subtract it.
    image_distances = np.linalg.norm(images[1:], axis=1)
    own_smooth_piece = 1.0 / (eta * np.sqrt(np.pi))
    madelung = float(
        np.sum(scipy.special.erfc(image_distances / (2.0 * eta)) / image_distances)
        + (2.0 * np.pi / area) * np.sum(g_weights)
        - own_smooth_piece
        - background)

    return EwaldSetup(lattice=lattice,
                      inverse_lattice=np.linalg.inv(lattice),
                      area=area,
                      eta=float(eta),
                      images=jnp.asarray(images),
                      g_vectors=jnp.asarray(g_vectors),
                      g_weights=jnp.asarray(g_weights),
                      background=float(background),
                      madelung=madelung)


# ---------------------------------------------------------------------------
#  The energy of one configuration
# ---------------------------------------------------------------------------

def wrap_into_supercell(setup, vectors):
    """Shift each vector by a lattice vector into the cell centred on the origin.

    pair_potential sums over a fixed window of images around the origin, so
    the separation it is given must be small.  This is not optional: the
    Metropolis walkers are never folded back into the cell, and they drift
    more than 5 supercells away within 3000 steps (measured in Stage 7).
    Shifting a separation by a lattice vector changes nothing physically,
    because the sum over images shifts with it.  For a slanted cell this is
    not exactly the minimum-image convention, and it does not need to be:
    the window of images is many cells wide.
    """
    fractional = vectors @ setup.inverse_lattice.T
    fractional = (fractional + 0.5) % 1.0 - 0.5
    return fractional @ setup.lattice.T


def pair_potential(setup, separation):
    """phi(r): the potential at r of a unit charge at the origin, all its
    images, and the neutralising background.

    Eq. (A9): Eq. (A6) summed over images, minus Eq. (A8), plus Eq. (A10).
    Infinite at r = 0, the charge's own position; coulomb_energy only ever
    evaluates it between two DIFFERENT electrons.
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

    positions: (N, 2).  strength = 1 gives a bare 1/r interaction; pass
    moire_interaction_strength() for the paper's (U/2) sum_{i != j} 1/r.
    """
    n_electrons = positions.shape[0]

    # First term of Eq. (A11): 1/2 sum over a != b.  Since phi(r) = phi(-r),
    # that equals summing each distinct pair a < b once.  Doing it this way
    # also means an electron is never paired with itself, where phi is
    # infinite: that value would be discarded anyway, but jax.grad would
    # turn the infinity into NaN.
    a, b = jnp.triu_indices(n_electrons, k=1)
    separations = wrap_into_supercell(setup, positions[a] - positions[b])
    pair_energies = jax.vmap(lambda r: pair_potential(setup, r))(separations)

    # Second term of Eq. (A11): each electron in the potential of its own images.
    self_image_energy = 0.5 * n_electrons * setup.madelung

    return strength * (jnp.sum(pair_energies) + self_image_energy)


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "stage03_slater"))
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "stage04_mcmc"))
    jax.config.update("jax_enable_x64", True)
    from slaternet import supercell_vectors, MOIRE_LATTICE
    from local_energy import moire_units

    energy_unit_meV, _ = moire_units()
    U = moire_interaction_strength(dielectric_constant=5.0)
    lattice = supercell_vectors(9)
    setup = make_ewald_setup(lattice)
    a1, a2 = MOIRE_LATTICE[:, 0], MOIRE_LATTICE[:, 1]
    grid = [(n, m) for n in range(3) for m in range(3)]
    sites = np.array([n * a1 + m * a2 for n, m in grid])          # the 9 moire sites

    print(f"9-site supercell:  area {setup.area:.6f} a_M^2,  eta {setup.eta:.6f} a_M")
    print(f"Madelung constant: xi_M = {setup.madelung:+.10f} / a_M")

    # A known answer.  One electron on every site is a perfect triangular
    # crystal; its energy per electron is -1.106102587 / r_s, with
    # r_s = sqrt(area per electron / pi)  (Bonsall and Maradudin 1977).
    r_s = np.sqrt(setup.area / 9 / np.pi)
    per_electron = float(coulomb_energy(setup, jnp.asarray(sites))) / 9
    print(f"triangular crystal, 9 electrons:  E/N * r_s = {per_electron * r_s:+.10f}"
          f"   (published -1.106102587)")

    # Filling 2/3 in the classical limit.  Put every electron on a site, a
    # minimum of the moire potential.  All 9 minima are equivalent, so the
    # moire energy is the same for every arrangement and only the Coulomb
    # energy decides; it depends only on relative positions, so it does not
    # matter where the minima sit within the cell.  Try all C(9, 6) = 84
    # arrangements.  The paper finds a honeycomb crystal at strong interaction
    # (Sec. V C, Fig. 6b): the 3 empty sites form one of the three
    # sqrt3 x sqrt3 sublattices.
    import itertools
    energy_of = jax.jit(lambda positions: coulomb_energy(setup, positions, strength=U))
    arrangements = list(itertools.combinations(range(9), 6))
    per_electron_meV = np.array([float(energy_of(jnp.asarray(sites[list(occupied)])))
                                 for occupied in arrangements]) / 6 * energy_unit_meV

    def is_honeycomb(occupied):
        empty = [grid[i] for i in range(9) if i not in occupied]
        return len({(n + m) % 3 for n, m in empty}) == 1     # all empty sites on one sublattice

    levels = np.unique(np.round(per_electron_meV, 9))
    lowest = [occupied for occupied, e in zip(arrangements, per_electron_meV) if e < levels[0] + 1e-6]
    print(f"\nfilling 2/3, classical limit, eps = 5 (U = {U:.6f}): all {len(arrangements)} "
          f"ways to put 6 electrons on the 9 sites")
    print(f"  lowest Coulomb energy  {levels[0]:+9.3f} meV per electron, reached by "
          f"{len(lowest)} arrangements, honeycombs: {all(is_honeycomb(o) for o in lowest)}")
    print(f"  next lowest            {levels[1]:+9.3f} meV per electron  "
          f"(gap {levels[1] - levels[0]:.3f} meV)")
    print(f"  highest                {levels[-1]:+9.3f} meV per electron")
