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

Stage 8 adds, at the end of this file, the same network with the settings of
the authors' released code (init_released, log_psi_released).
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


def orbitals_from_hidden(projections: jax.Array, h: jax.Array) -> jax.Array:
    """Eq. (22): hidden states (N, width) -> orbital matrices (D, N, N), complex.

    Element [m, i, j] = orbital j of determinant m, evaluated at electron i.
    Shared with Stage 5, which changes only how h is computed.
    """
    real_part = jnp.einsum("mjd,id->mij", projections[:, :, 0, :], h)
    imag_part = jnp.einsum("mjd,id->mij", projections[:, :, 1, :], h)
    return real_part + 1j * imag_part


def orbital_matrices(params: dict, positions: jax.Array, G: jax.Array) -> jax.Array:
    """SlaterNet's orbital matrices, shape (D, N, N)."""
    return orbitals_from_hidden(params["proj"], hidden_states(params, positions, G))


class LogPsi(NamedTuple):
    log_abs: jax.Array   # log |Psi|
    phase:   jax.Array   # arg Psi, radians


def sum_of_determinants(matrices: jax.Array) -> LogPsi:
    """Eq. (23): Psi = sum_m det(matrices[m]), returned as log|Psi| and arg Psi.

    matrices: (D, N, N) complex orbital matrices.  Shared with Stage 5.

    Why never form Psi?  A determinant is a sum of N! products of N numbers;
    for entries of typical size s, |det| ~ sqrt(N!) s^N.  At the paper's Table I
    size (18 electrons) we measured |Psi| ~ 1e-41 to 1e-34 at initialisation,
    below float32's smallest normal number (~1e-38).  VMC never needs Psi
    itself anyway: Metropolis uses |Psi'/Psi|^2 = exp(2 (log|Psi'| - log|Psi|)),
    and the local energy uses derivatives of log Psi.

    log|Psi| = -inf exactly where Psi vanishes, never NaN.
    """
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


def log_psi(params: dict, positions: jax.Array, G: jax.Array) -> LogPsi:
    """SlaterNet's wave function: log|Psi| and arg Psi at one configuration."""
    return sum_of_determinants(orbital_matrices(params, positions, G))


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


# ---------------------------------------------------------------------------
#  Stage 8: the same network with the settings of the authors' released code
# ---------------------------------------------------------------------------
#
# Everything above follows the paper (Table II) and is unchanged.  Stage 8
# trains with our own code but with the settings of the authors' released
# moire config, the settings their reference run used, so that the two runs
# can be compared.  The differences:
#
#                          above (paper)          released code (below)
#   inputs per electron    4 periodic features    the same 4, plus its spin (+1)
#   activation             tanh                   GELU, tanh approximation
#   perceptrons per layer  1                      2, each followed by GELU
#   after each residual    nothing                layer norm
#   layers, width          3, 64                  4, 64
#   initial weights        0.1 * normal           normal / sqrt(inputs); biases normal
#
# Weights are stored as (inputs, outputs) and used as x @ w + b, the authors'
# layout.  learn/stage08_own_code/test_networks.py checks that these networks
# ARE the authors' networks: same function and same derivatives with their
# trained weights, same initial distribution, and the same layers for KFAC.
#
# Added for 18 electrons (2026-10-30): determinants="slogdet" adds the
# determinants with slogdet, as the authors' code does, instead of with
# sum_of_determinants above.  Only then can folx compute the kinetic energy
# fast (learn/stage08_own_code/speed_study/speed_study_report.pdf).  The
# default "det" is what every earlier run used.

@dataclass(frozen=True)
class ReleasedArchitecture:
    """SlaterNet sizes in the authors' released moire config."""
    n_electrons: int
    n_determinants: int = 1
    n_layers: int = 4
    width: int = 64
    perceptrons_per_layer: int = 2
    determinants: str = "det"       # "det": sum_of_determinants;  "slogdet": sum_of_determinants_slogdet


def dense_init(key: jax.Array, n_in: int, n_out: int, bias: bool = True) -> dict:
    """One dense layer, initialised as in the authors' code (network_blocks.py):
    weights normal / sqrt(n_in), biases normal."""
    key_w, key_b = jax.random.split(key)
    layer = {"w": jax.random.normal(key_w, (n_in, n_out)) / np.sqrt(n_in)}
    if bias:
        layer["b"] = jax.random.normal(key_b, (n_out,))
    return layer


def dense(x: jax.Array, layer: dict) -> jax.Array:
    """x @ w + b, or x @ w for a layer without bias."""
    y = x @ layer["w"]
    if "b" in layer:
        y = y + layer["b"]
    return y


def layer_norm_init(width: int) -> dict:
    return {"scale": jnp.ones(width), "offset": jnp.zeros(width)}


def layer_norm(norm: dict, h: jax.Array) -> jax.Array:
    """Each electron's hidden state to mean 0 and variance 1, then a learned
    scale and offset.  Same operations in the same order as the authors'
    make_layer_norm, epsilon 1e-5 included."""
    mean = jnp.mean(h, axis=-1, keepdims=True)
    variance = jnp.var(h, axis=-1, keepdims=True)
    inverse = norm["scale"] * jax.lax.rsqrt(variance + 1e-5)
    return inverse * (h - mean) + norm["offset"]


def perceptrons(layers: list, h: jax.Array) -> jax.Array:
    """The perceptrons of one layer, each followed by GELU.  jax.nn.gelu uses
    the tanh approximation unless told otherwise; the authors call it that way."""
    for layer in layers:
        h = jax.nn.gelu(dense(h, layer), approximate=True)
    return h


def released_inputs(positions: jax.Array, G: jax.Array) -> jax.Array:
    """(N,2) -> (N,5): the 4 periodic features of Eq. (7), then the spin.

    All electrons have spin +1 here, so the fifth input is a constant: it acts
    as a bias of the embedding.
    """
    features = periodic_features(positions, G)
    spin = jnp.ones_like(features[:, :1])
    return jnp.concatenate([features, spin], axis=1)


def released_orbitals(orbital_layer: dict, h: jax.Array, n_determinants: int) -> jax.Array:
    """Hidden states (N, width) -> orbital matrices (D, N, N), complex.

    One dense layer gives 2 N D numbers per electron: in the authors' packing
    the even ones are real parts, the odd ones imaginary parts.  Element
    [m, i, j] is orbital j of determinant m at electron i, as in
    orbitals_from_hidden.
    """
    n = h.shape[0]
    out = dense(h, orbital_layer)                    # (N, 2 N D)
    values = out[:, 0::2] + 1j * out[:, 1::2]        # (N, N D)
    return values.reshape(n, n_determinants, n).transpose(1, 0, 2)


def init_released(key: jax.Array, arch: ReleasedArchitecture) -> dict:
    """SlaterNet parameters with the released settings."""
    key_embed, key_orbital, *layer_keys = jax.random.split(key, arch.n_layers + 2)
    n_orbital_outputs = 2 * arch.n_electrons * arch.n_determinants
    params = {
        "embed": dense_init(key_embed, 5, arch.width, bias=False),
        "layers": [],
        "orbital": dense_init(key_orbital, arch.width, n_orbital_outputs, bias=False),
    }
    for layer_key in layer_keys:
        perceptron_keys = jax.random.split(layer_key, arch.perceptrons_per_layer)
        params["layers"].append({
            "perceptrons": [dense_init(k, arch.width, arch.width) for k in perceptron_keys],
            "norm": layer_norm_init(arch.width),
        })
    return params


def hidden_states_released(params: dict, positions: jax.Array, G: jax.Array) -> jax.Array:
    """(N,2) -> (N,width).  Still every electron on its own: Hartree-Fock with D = 1."""
    h = dense(released_inputs(positions, G), params["embed"])
    for layer in params["layers"]:
        h = layer_norm(layer["norm"], h + perceptrons(layer["perceptrons"], h))
    return h


def sum_of_determinants_slogdet(matrices: jax.Array) -> LogPsi:
    """The same as sum_of_determinants, but each determinant comes from slogdet.

    Why a second version.  folx, the library that computes the kinetic energy
    fast (Stage 4, kinetic_energy_folx), has a hand-written rule for the
    second derivatives of slogdet and none for jnp.linalg.det.  With det it
    falls back to a full table of second derivatives over every matrix
    entry: at 18 electrons and 1024 walkers about 39 GiB, more than the GPU
    has.  With slogdet the whole local energy needs 1.4 GiB
    (learn/stage08_own_code/speed_study/speed_study_report.pdf).

    The price, and why sum_of_determinants uses det.  If ONE determinant is
    exactly zero while the others are not, Psi is smooth, but here the chain
    rule multiplies d log|det A| = tr(A^-1 dA) = infinity by det A = 0, and
    the derivatives are NaN.  test_slaternet.py shows it.  Random walkers
    essentially never sit exactly there; the authors' code (FermiNet's
    logdet_matmul) adds its determinants the same way.

    The steps are those of sum_of_determinants: one common shift, found from
    the values only, so that nothing over- or underflows; then add.
    """
    sign, log_abs = jnp.linalg.slogdet(matrices)       # sign: a complex number of size 1
    largest = jax.lax.stop_gradient(jnp.max(log_abs))
    shift = jnp.where(jnp.isfinite(largest), largest, 0.0)
    total = jnp.sum(sign * jnp.exp(log_abs - shift))    # = Psi * exp(-shift)
    return LogPsi(log_abs=shift + jnp.log(jnp.abs(total)),
                  phase=jnp.angle(total))


def determinant_sum(name: str):
    """The function that adds the determinants: "det" or "slogdet"."""
    if name == "det":
        return sum_of_determinants
    if name == "slogdet":
        return sum_of_determinants_slogdet
    raise ValueError(f"determinants must be 'det' or 'slogdet', not {name!r}")


def log_psi_released(params: dict, positions: jax.Array, G: jax.Array,
                     arch: ReleasedArchitecture) -> LogPsi:
    """SlaterNet with the released settings: log|Psi| and arg Psi."""
    h = hidden_states_released(params, positions, G)
    add = determinant_sum(arch.determinants)
    return add(released_orbitals(params["orbital"], h, arch.n_determinants))


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
