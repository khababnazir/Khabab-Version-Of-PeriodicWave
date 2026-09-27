"""
Stage 3 -- the many-electron wave function (SlaterNet).

Paper:   Geier et al., PRB 112, 045119 (2025), Sec. III B, Eqs. (11)-(14), (22)-(23)
Authors: external/PeriodicWave/periodicwave/SlaterNet.py and networks.make_orbitals

    positions (N,2) -> hidden states (N,width) -> orbital matrices (D,N,N) -> log Psi

Key ideas
  1. Rows are electrons, columns are orbitals.  Swapping two electrons swaps two
     rows, which flips the determinant's sign: antisymmetry is exact for every
     value of the parameters.
  2. We return log|Psi| and the phase, never Psi itself (see log_psi).
  3. With ONE determinant this is Hartree-Fock: no electron ever sees another.
     With several it is not -- see Architecture.n_determinants.

Scope: spin-polarised electrons (all spins equal), as in the paper.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
#  Geometry: plain NumPy, fixed at setup, never differentiated
# ---------------------------------------------------------------------------

# C6-symmetric clusters from PeriodicWave's pbc/lattices.py, keyed by site count.
CLUSTER_MATRICES = {
    3:  [[1, 1], [2, -1]],
    9:  [[3, 0], [0, 3]],
    12: [[2, 2], [4, -2]],
    21: [[1, 4], [5, -1]],
    27: [[3, 3], [6, -3]],
    36: [[6, 0], [0, 6]],
}

# Primitive moire lattice in units of a_M.  Lattice vectors are the COLUMNS.
MOIRE_LATTICE = np.array([[1.0, -0.5],
                          [0.0, np.sqrt(3.0) / 2.0]])


def supercell_vectors(n_sites: int) -> np.ndarray:
    """Supercell vectors L1, L2 as the columns of a 2x2 matrix."""
    if n_sites not in CLUSTER_MATRICES:
        raise ValueError(f"no cluster for {n_sites} sites; have {sorted(CLUSTER_MATRICES)}")
    M = np.array(CLUSTER_MATRICES[n_sites])
    a1, a2 = MOIRE_LATTICE[:, 0], MOIRE_LATTICE[:, 1]
    return np.column_stack([M[0, 0] * a1 + M[1, 0] * a2,
                            M[0, 1] * a1 + M[1, 1] * a2])


def reciprocal_vectors(lattice: np.ndarray) -> np.ndarray:
    """G1, G2 as the ROWS, so that G_i . L_j = 2 pi delta_ij."""
    return 2.0 * np.pi * np.linalg.inv(lattice)


# ---------------------------------------------------------------------------
#  Network
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Architecture:
    """Sizes that fix the shape of the model.  Not trainable."""
    n_electrons: int
    # 1 determinant  = Hartree-Fock.  This is the paper's SlaterNet.
    # D > 1 determinants of independent orbitals is already a small
    # multi-determinant expansion that captures some correlation, so it is no
    # longer a Hartree-Fock reference.  The self-attention network uses D = 4.
    n_determinants: int = 1
    n_layers: int = 3
    width: int = 64
    init_scale: float = 0.1


def init_parameters(key: jax.Array, arch: Architecture) -> dict:
    """Variational parameters as a plain nested dict (a JAX 'pytree')."""
    k_embed, k_proj, *k_layers = jax.random.split(key, arch.n_layers + 2)
    s, w = arch.init_scale, arch.width

    # One (real, imaginary) pair of projection vectors per determinant and
    # orbital, Eq. (22).  Shape (D, N, 2, width).
    proj_shape = (arch.n_determinants, arch.n_electrons, 2, w)

    return {
        "embed":  s * jax.random.normal(k_embed, (w, 4)),                # W0, Eq. (8)
        "layers": [(s * jax.random.normal(k, (w, w)), jnp.zeros(w))      # W^l, b^l, Eq. (9)
                   for k in k_layers],
        "proj":   s * jax.random.normal(k_proj, proj_shape),             # Eq. (22)
    }


def periodic_features(positions: jax.Array, G: jax.Array) -> jax.Array:
    """Eq. (7): (N,2) -> (N,4), sin and cos of the phases G_i . r.

    Difference from the authors' code: they append each electron's spin as a
    fifth input.  For spin-polarised electrons that is a constant, so it acts
    as a bias on the embedding.  We follow the paper -- four inputs, no bias --
    which is also what its Appendix C parameter count assumes.
    """
    phases = positions @ jnp.asarray(G).T
    return jnp.concatenate([jnp.sin(phases), jnp.cos(phases)], axis=-1)


def hidden_states(params: dict, positions: jax.Array, G: jax.Array) -> jax.Array:
    """Eqs. (8)-(9): (N,2) -> (N,width).

    Every electron goes through the same weights, independently.  Nothing lets
    electron i affect electron j's orbital.  With one determinant, that absence
    IS the independent-electron (Hartree-Fock) approximation.  Stage 5 adds
    attention, the one change that lets the electrons see each other.
    """
    h = periodic_features(positions, G) @ params["embed"].T
    for weight, bias in params["layers"]:
        h = h + jnp.tanh(h @ weight.T + bias)
    return h


def orbital_matrices(params: dict, positions: jax.Array, G: jax.Array) -> jax.Array:
    """Eq. (22): complex array of shape (D, N, N).

    Element [m, i, j] = orbital j of determinant m, evaluated at electron i.
    """
    h = hidden_states(params, positions, G)
    real_part = jnp.einsum("mjd,id->mij", params["proj"][:, :, 0, :], h)
    imag_part = jnp.einsum("mjd,id->mij", params["proj"][:, :, 1, :], h)
    return real_part + 1j * imag_part


class LogPsi(NamedTuple):
    log_abs: jax.Array   # log |Psi|
    phase:   jax.Array   # arg Psi, radians


def log_psi(params: dict, positions: jax.Array, G: jax.Array) -> LogPsi:
    """Eq. (23), Psi = sum_m det[phi^m_j(r_i)], returned as log|Psi| and arg Psi.

    Why never form Psi?  A determinant is a sum of N! products of N numbers;
    for entries of typical size s, |det| ~ sqrt(N!) s^N.  At the paper's Table I
    size (18 electrons) we measured |Psi| ~ 1e-41 to 1e-34 at initialisation,
    below float32's smallest normal number (~1e-38).  VMC never needs Psi
    itself anyway: Metropolis uses |Psi'/Psi|^2 = exp(2 (log|Psi'| - log|Psi|)),
    and the local energy uses derivatives of log Psi.

    log|Psi| = -inf exactly where Psi vanishes, never NaN.
    """
    matrices = orbital_matrices(params, positions, G)
    n = matrices.shape[-1]

    # Step 1 -- find one common scale, from values only (no gradient flows here).
    # slogdet reports each |det| as a logarithm, so nothing over- or underflows
    # while we look.  If every determinant is exactly zero, largest = -inf;
    # using 0 instead avoids (-inf) - (-inf) = NaN.
    _, log_abs = jnp.linalg.slogdet(jax.lax.stop_gradient(matrices))
    largest = jnp.max(log_abs)
    shift = jnp.where(jnp.isfinite(largest), largest, 0.0)

    # Step 2 -- rescale every matrix by the same factor c = exp(-shift / n), so
    # det(c A) = exp(-shift) det(A): the largest determinant now has size 1 and
    # nothing is out of floating-point range.  Then add plain determinants.
    #
    # Why plain det rather than slogdet here?  The derivative of slogdet is
    # d log det A = tr(A^-1 dA), which is infinite when A is singular.  If ONE
    # determinant vanishes while the others do not, Psi is perfectly smooth,
    # but the chain rule multiplies that infinity by the vanishing determinant:
    # 0 * inf = NaN.  (Nearly singular is also bad: huge times tiny, precision
    # lost.)  The derivative of det itself, d det A = tr(adj(A) dA), stays
    # finite for singular A, and jnp.linalg.det differentiates that way.
    #
    # Known limitation, measured in JAX 0.8.0: jnp.linalg.det's SECOND
    # derivative is NaN at matrices of rank <= N-2 when N >= 4 (e.g. a
    # determinant that is identically zero).  Rank N-1 -- the generic singular
    # case, a configuration on one determinant's nodal surface -- is correct
    # at first and second order.  test_slaternet.py checks both.
    scale = jnp.exp(-shift / n)
    total = jnp.sum(jnp.linalg.det(scale * matrices))      # = Psi * exp(-shift)

    # Psi = 0 has no phase.  jnp.angle(0) = 0 is our documented placeholder.
    return LogPsi(log_abs=shift + jnp.log(jnp.abs(total)),
                  phase=jnp.angle(total))


# ---------------------------------------------------------------------------
#  What the VMC loop will call
# ---------------------------------------------------------------------------

@jax.jit
def log_abs_psi(params: dict, positions: jax.Array, G: jax.Array) -> jax.Array:
    """log|Psi| alone -- a real scalar, so jax.grad accepts it."""
    return log_psi(params, positions, G).log_abs


# d log|Psi| / d parameters: the MAGNITUDE half of the log-derivative only.
# For a complex Psi the exact energy gradient needs the full complex
#     d(log Psi)/d theta = d log|Psi|/d theta + i * d(phase)/d theta,
# which Stage 6 builds.  (PeriodicWave's train.py switches to that complex
# log-derivative whenever the network is complex.)
grad_log_abs_wrt_params = jax.jit(jax.grad(log_abs_psi, argnums=0))

# d log|Psi| / d positions   -> one ingredient of the kinetic energy.
grad_log_abs_wrt_positions = jax.jit(jax.grad(log_abs_psi, argnums=1))

# Many MCMC walkers at once, as one fused kernel instead of a Python loop.
log_abs_psi_batched = jax.jit(jax.vmap(log_abs_psi, in_axes=(None, 0, None)))


def random_walkers(key: jax.Array, n_walkers: int, n_electrons: int,
                   lattice: np.ndarray) -> jax.Array:
    """Uniform positions inside the supercell: fractional -> cartesian."""
    frac = jax.random.uniform(key, (n_walkers, n_electrons, 2))
    return frac @ jnp.asarray(lattice).T


if __name__ == "__main__":
    import time

    lattice = supercell_vectors(9)
    G = reciprocal_vectors(lattice)

    hf = Architecture(n_electrons=6)                        # 9 sites, 2/3 filling
    params = init_parameters(jax.random.key(0), hf)
    positions = random_walkers(jax.random.key(1), 1, hf.n_electrons, lattice)[0]

    out = log_psi(params, positions, G)
    print(f"log|Psi| = {out.log_abs:+.6f}    phase = {out.phase:+.6f} rad")
    print(f"orbital matrices shape = {orbital_matrices(params, positions, G).shape}")

    swapped = positions.at[jnp.array([0, 1])].set(positions[jnp.array([1, 0])])
    out_swapped = log_psi(params, swapped, G)
    print(f"swap two electrons:  d log|Psi| = {float(out_swapped.log_abs - out.log_abs):+.2e}"
          f"   d phase = {float(out_swapped.phase - out.phase):+.6f} rad")

    n_par = sum(x.size for x in jax.tree.leaves(params))
    print(f"n_parameters (1 det)  = {n_par}")

    walkers = random_walkers(jax.random.key(2), 1024, hf.n_electrons, lattice)
    log_abs_psi_batched(params, walkers, G).block_until_ready()        # compile
    t0 = time.perf_counter()
    for _ in range(10):
        log_abs_psi_batched(params, walkers, G).block_until_ready()
    print(f"1024 walkers, compiled: {(time.perf_counter() - t0) / 10 * 1e3:.2f} ms/call")

    grads = grad_log_abs_wrt_params(params, positions, G)
    print("gradient tree matches parameter tree:",
          jax.tree.structure(grads) == jax.tree.structure(params))

    # How big is Psi really, at the paper's Table I size?
    lattice27 = supercell_vectors(27)
    G27 = reciprocal_vectors(lattice27)
    table1 = Architecture(n_electrons=18)
    p27 = init_parameters(jax.random.key(3), table1)
    w27 = random_walkers(jax.random.key(4), 1024, table1.n_electrons, lattice27)
    la = log_abs_psi_batched(p27, w27, G27)
    print(f"18 electrons / 27 sites, 1024 walkers at init: log10|Psi| in "
          f"[{float(la.min()) / np.log(10):.1f}, {float(la.max()) / np.log(10):.1f}]")
