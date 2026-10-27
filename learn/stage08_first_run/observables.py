"""
Stage 8 -- what a trained wave function says about WHERE the electrons are.

The energy is a single number.  To see which state the network has found --
a Fermi liquid spread over all moire sites, or a generalized Wigner crystal
that fills only some of them -- we look at the electron positions sampled
from its |Psi|^2.  This file turns batches of those samples into the
quantities of the paper's Fig. 6, plus three more that are easier to read
in a small 9-site cell.

  density            n(r): where the electrons are.                Fig. 6(a, b)
  pair correlation   g(dr): where the OTHER electrons are, seen
                     from one electron.                             Fig. 6(c, d)
  pair distances     g(r): the same, averaged over directions.
  site occupations   each electron is assigned to its nearest moire
                     site; which of the 9 sites are filled?
  structure factor   S(q) = <|rho_q|^2> / N.

Why the pair correlation and not only the density.  In a 9-site cell at
nu = 2/3 there are three equivalent honeycomb crystals (Stage 7).  If the
wave function is an equal mixture of all three, every site is filled 2/3 of
the time and the density looks uniform, crystal or not.  The pair
correlation still shows the crystal: it asks where the other electrons are
GIVEN that one sits here.

Conventions.  Positions R have shape (walkers, electrons, 2), in units of
a_M.  The supercell is spanned by the COLUMNS of `supercell` (3 a1 and 3 a2,
as in the authors' code).  A position r has fractional coordinates
s = supercell^-1 r, and the supercell is 0 <= s < 1.  Histograms are taken in
s, so all bins have the same area, area(supercell) / bins^2.

The functions measured during the run (JAX, top half) return SUMS over the
walkers, never means: the evaluation adds them up over thousands of steps,
and the functions at the bottom (NumPy) divide by the number of samples once.
"""
import itertools

import numpy as np
import jax
import jax.numpy as jnp

SITES_PER_SIDE = 3          # the 3 x 3 supercell of the paper's Fig. 5
N_SITES = SITES_PER_SIDE ** 2
SOME_SITE_DOUBLY_OCCUPIED = 2 ** N_SITES      # pattern code, see pattern_counts


# ------------------------------------------------------------------ geometry

def moire_sites(supercell):
    """The 9 moire lattice points in the supercell, at s = (i/3, j/3).

    They are the minima of the moire potential for phi = 45 degrees
    (checked in test_observables.py), i.e. where an electron prefers to sit.
    """
    fractions = np.array([(i, j) for i in range(SITES_PER_SIDE)
                          for j in range(SITES_PER_SIDE)]) / SITES_PER_SIDE
    return fractions @ np.asarray(supercell).T


def minimum_image(d, supercell):
    """The shortest vector equivalent to d under translations by the supercell.

    Reducing the fractional coordinates to [-1/2, 1/2) is not enough when the
    cell vectors meet at 120 degrees: near the corners of that parallelogram
    a neighbouring image is closer.  So the 9 nearest images are compared and
    the shortest is kept.
    """
    supercell = jnp.asarray(supercell, dtype=d.dtype)
    s = d @ jnp.linalg.inv(supercell).T
    s = s - jnp.round(s)
    shifts = jnp.array(list(itertools.product((-1, 0, 1), repeat=2)), dtype=d.dtype)
    images = (s[..., None, :] + shifts) @ supercell.T             # (..., 9, 2)
    shortest = jnp.argmin(jnp.sum(images ** 2, axis=-1), axis=-1)
    return jnp.take_along_axis(images, shortest[..., None, None], axis=-2)[..., 0, :]


def supercell_q_vectors(supercell, q_max):
    """All nonzero wave vectors q = B m (m integer) of the supercell with |q| <= q_max.

    B = 2 pi (supercell^-1)^T, so exp(i q.r) is periodic on the supercell.
    These are the only q at which S(q) is defined in a periodic cell.
    """
    B = 2 * np.pi * np.linalg.inv(np.asarray(supercell)).T
    m_max = int(np.ceil(q_max / np.min(np.linalg.norm(B, axis=0)))) + 1
    ms = np.array([m for m in itertools.product(range(-m_max, m_max + 1), repeat=2) if m != (0, 0)])
    qs = ms @ B.T
    keep = np.linalg.norm(qs, axis=1) <= q_max + 1e-9
    return qs[keep], ms[keep]


def honeycomb_codes(supercell):
    """Pattern codes (see pattern_counts) of the three honeycomb crystals.

    Six electrons on nine sites leave three sites empty.  The arrangement is
    a honeycomb exactly when the empty sites form a triangular lattice of
    spacing sqrt(3) a_M -- every pair of empty sites sqrt(3) apart.
    """
    sites = moire_sites(supercell)
    codes = []
    for empty in itertools.combinations(range(N_SITES), 3):
        distances = [float(jnp.linalg.norm(minimum_image(jnp.asarray(sites[a] - sites[b]), supercell)))
                     for a, b in itertools.combinations(empty, 2)]
        if np.allclose(distances, np.sqrt(3)):
            filled = [i for i in range(N_SITES) if i not in empty]
            codes.append(sum(2 ** i for i in filled))
    return codes


# --------------------------------------------- measured during the run (JAX)

def _histogram_2d(s, bins):
    """Counts of fractional points s (shape (..., 2), values in [0, 1)) on a bins x bins grid."""
    index = jnp.clip(jnp.floor(s * bins).astype(jnp.int32), 0, bins - 1)
    flat = index[..., 0] * bins + index[..., 1]
    return jnp.bincount(flat.ravel(), length=bins * bins).reshape(bins, bins)


def density_counts(R, supercell, bins):
    """How many electrons fall into each bin of the supercell, summed over walkers."""
    s = R @ jnp.linalg.inv(jnp.asarray(supercell, dtype=R.dtype)).T
    return _histogram_2d(s - jnp.floor(s), bins)


def _ordered_pairs(n_electrons):
    """Index arrays (i, j) of all ordered pairs of different electrons."""
    i, j = np.nonzero(~np.eye(n_electrons, dtype=bool))
    return i, j


def pair_counts(R, supercell, bins):
    """Histogram of r_j - r_i over all ordered pairs i != j, summed over walkers.

    The displacement is reduced to fractional coordinates in [-1/2, 1/2), so
    the reference electron sits at the CENTRE of the map, and then shifted
    by +1/2 for binning.
    """
    i, j = _ordered_pairs(R.shape[1])
    s = (R[:, j] - R[:, i]) @ jnp.linalg.inv(jnp.asarray(supercell, dtype=R.dtype)).T
    return _histogram_2d(s - jnp.round(s) + 0.5, bins)


def pair_distance_counts(R, supercell, r_max, n_bins):
    """Histogram of the minimum-image distance |r_j - r_i| over pairs i < j, for r < r_max."""
    i, j = np.triu_indices(R.shape[1], k=1)
    r = jnp.linalg.norm(minimum_image(R[:, j] - R[:, i], supercell), axis=-1)
    index = jnp.minimum(jnp.floor(r / r_max * n_bins).astype(jnp.int32), n_bins)   # n_bins = beyond r_max
    return jnp.bincount(index.ravel(), length=n_bins + 1)[:n_bins]


def site_occupations(R, supercell):
    """Number of electrons whose nearest moire site is site k: shape (walkers, 9)."""
    sites = jnp.asarray(moire_sites(supercell), dtype=R.dtype)
    d = minimum_image(R[:, :, None, :] - sites, supercell)            # (walkers, electrons, sites, 2)
    nearest = jnp.argmin(jnp.sum(d ** 2, axis=-1), axis=-1)            # (walkers, electrons)
    return jax.nn.one_hot(nearest, N_SITES, dtype=jnp.int32).sum(axis=1)


def pattern_counts(occupations):
    """How often each filling pattern occurs, summed over walkers.

    A pattern with at most one electron per site gets the code
    sum_k n_k 2^k (one bit per site, 0 ... 511).  Every configuration with
    two or more electrons nearest the same site gets the single code 512.
    """
    one_per_site = jnp.all(occupations <= 1, axis=1)
    code = jnp.sum(occupations * (2 ** jnp.arange(N_SITES)), axis=1)
    code = jnp.where(one_per_site, code, SOME_SITE_DOUBLY_OCCUPIED)
    return jnp.bincount(code, length=SOME_SITE_DOUBLY_OCCUPIED + 1)


def density_fourier(R, qs):
    """rho_q = sum_j exp(-i q.r_j), one value per walker and q: shape (walkers, n_q)."""
    phases = jnp.einsum("wnd,qd->wnq", R, jnp.asarray(qs, dtype=R.dtype))
    return jnp.sum(jnp.exp(-1j * phases), axis=1)


def measure(R, supercell, qs, bins, r_max, n_radial_bins):
    """Everything above for one batch of walkers, as a dict of sums over the walkers."""
    occupations = site_occupations(R, supercell)
    rho = density_fourier(R, qs)
    return dict(
        density=density_counts(R, supercell, bins),
        pairs=pair_counts(R, supercell, bins),
        pair_distances=pair_distance_counts(R, supercell, r_max, n_radial_bins),
        patterns=pattern_counts(occupations),
        occupation=occupations.sum(axis=0),
        occupation_products=occupations.T @ occupations,      # sum over walkers of n_k n_l
        rho_q=rho.sum(axis=0),
        rho_q_squared=jnp.sum(jnp.abs(rho) ** 2, axis=0),
    )


# --------------------------------------------- normalised afterwards (NumPy)

def supercell_area(supercell):
    return abs(np.linalg.det(np.asarray(supercell)))


def density_map(counts, n_configurations, supercell):
    """n(r) in electrons per a_M^2.  Integrates to the number of electrons."""
    bin_area = supercell_area(supercell) / counts.size
    return counts / (n_configurations * bin_area)


def pair_correlation_map(counts, n_configurations, n_electrons, supercell):
    """g(dr): 1 where the other electrons are where they would be if uncorrelated.

    Normalised by what N(N-1) independent, uniformly spread pairs would give,
    so g -> 1 without correlations and g = 0 where the other electrons never are.
    """
    pairs_per_configuration = n_electrons * (n_electrons - 1)
    expected_per_bin = n_configurations * pairs_per_configuration / counts.size
    return counts / expected_per_bin


def radial_pair_correlation(counts, n_configurations, n_electrons, supercell, r_max):
    """g(r) from pair_distance_counts, normalised the same way; valid up to r_max."""
    edges = np.linspace(0.0, r_max, len(counts) + 1)
    shell_area = np.pi * (edges[1:] ** 2 - edges[:-1] ** 2)
    pairs_per_configuration = n_electrons * (n_electrons - 1) / 2
    expected = n_configurations * pairs_per_configuration * shell_area / supercell_area(supercell)
    return 0.5 * (edges[1:] + edges[:-1]), counts / expected


def structure_factor(rho_q_sum, rho_q_squared_sum, n_configurations, n_electrons):
    """S(q) = <|rho_q|^2> / N, and its connected part (<|rho_q|^2> - |<rho_q>|^2) / N.

    The difference is |<rho_q>|^2 / N: a density modulation that the
    samples show directly (the moire potential at its own wave vectors, or a
    crystal that the sampling has locked into one of its patterns).
    """
    mean_rho = rho_q_sum / n_configurations
    mean_rho_squared = rho_q_squared_sum / n_configurations
    return mean_rho_squared / n_electrons, (mean_rho_squared - np.abs(mean_rho) ** 2) / n_electrons
