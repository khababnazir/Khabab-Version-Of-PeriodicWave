"""
Stage 1 -- the one-body orbital network of Geier, Nazaryan, Zaklama & Fu,
Phys. Rev. B 112, 045119 (2025).  Implements Eqs. (5)-(10): the map from one
electron position r in R^2 to one complex orbital amplitude phi(r) in C.

This is the innermost piece of the entire paper.  Everything later wraps it:
Stage 3 makes N of these into a Slater determinant, Stage 5 lets the N
electron streams talk to each other through self-attention, and the VMC loop
optimises the parameters.  Nothing in this file knows any physics except one
fact -- that the supercell is periodic.
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

# ---------------------------------------------------------------------------
# Geometry.  Plain NumPy on purpose: this is fixed setup, never differentiated,
# so it does not need to live inside JAX.
# ---------------------------------------------------------------------------

# 9-site supercell = 3x3 moire unit cells, in units of the moire lattice
# constant a_M.  Lifted from PeriodicWave's pbc/lattices.py, which for
# num_sites=9 falls through to the M = diag(3,3) branch.
# CONVENTION, and it is easy to get backwards: the lattice vectors are the
# COLUMNS of L.  L1 = L[:, 0] = (3, 0);  L2 = L[:, 1] = (-1.5, 2.598...).
L_9SITE = np.array([[3.0, -1.5],
                    [0.0, 3.0 * np.sqrt(3.0) / 2.0]])


def reciprocal(L: np.ndarray) -> np.ndarray:
    """Reciprocal supercell vectors, returned as the ROWS of the result.

    Defined by G_i . L_j = 2*pi*delta_ij, which is exactly G = 2*pi*L^{-1}
    when the L_j are the columns of L.  That identity is the whole reason the
    feature map below is exactly periodic rather than approximately periodic.
    """
    return 2.0 * np.pi * np.linalg.inv(L)


# ---------------------------------------------------------------------------
# The network
# ---------------------------------------------------------------------------

def feature(r, G):
    """Eq. (7): periodic coordinates.  (2,) -> (4,)

    The network never sees r.  A wave function in a periodic solid must obey
    Psi(..., r_i + L, ...) = Psi(..., r_i, ...) for every supercell vector L.
    Feed raw coordinates in and the network can only ever *approximate* that,
    and it will approximate it worst exactly at the cell boundary, which is
    where the physics of a crystal lives.

    Feeding sin and cos of G.r instead makes periodicity exact by
    construction: shifting r by L_j changes G_i.r by exactly 2*pi*delta_ij,
    and sin/cos cannot tell the difference.  The periodicity is therefore a
    property of the parameterisation, not something the optimiser has to
    discover and can lose again.

    There is a second, subtler benefit.  Two points on opposite sides of the
    cell boundary are physical neighbours but have wildly different raw
    coordinates.  This map sends them to neighbouring feature vectors, so the
    network's smoothness in feature space matches smoothness in real space.
    """
    gr = jnp.asarray(G) @ r                              # (2,) phases G1.r, G2.r
    return jnp.concatenate([jnp.sin(gr), jnp.cos(gr)])   # (4,)


def init_params(key, d_L: int = 64, n_layers: int = 3,
                d_feat: int = 4, scale: float = 0.1):
    """Allocate the variational parameters theta.

    Returned as a plain nested dict -- what JAX calls a pytree.  Note what is
    absent: there is no class, no nn.Module, no self.W.  Parameters are
    ordinary data that you hand to the function explicitly.

    That is not a stylistic preference, it is what makes
    jax.grad(f)(params, ...) trustworthy: with no hidden mutable state, there
    is nothing for the gradient to silently miss, and the same params can be
    used by two devices at once without either one corrupting the other.

    The RNG is explicit for the same reason.  jax.random.split turns one key
    into many, so every weight gets independent randomness and the whole
    initialisation is reproducible from the single key you passed in.
    """
    keys = jax.random.split(key, 2 * n_layers + 3)
    params = {
        "W0": scale * jax.random.normal(keys[0], (d_L, d_feat)),   # Eq. (8)
        "layers": [],
        "w0": scale * jax.random.normal(keys[1], (d_L,)),          # Eq. (10)
        "w1": scale * jax.random.normal(keys[2], (d_L,)),
    }
    for l in range(n_layers):
        params["layers"].append((
            scale * jax.random.normal(keys[3 + 2 * l], (d_L, d_L)),  # W^(l+1)
            jnp.zeros((d_L,)),                                       # b^(l+1)
        ))
    return params


def orbital(params, r, G):
    """Eqs. (8)-(10).  One position -> one complex amplitude.

    The residual form in Eq. (9), h <- h + tanh(W h + b), rather than the
    textbook h <- tanh(W h + b), matters twice over.

    Representationally: with the skip term, the identity map is free -- set
    W = 0 and the layer passes h through untouched.  A stack of L such layers
    can therefore always express anything a stack of L-1 can, so making the
    network deeper cannot make it worse.  Without the skip, every layer is
    forced to transform its input, and depth becomes a gamble.

    Numerically: tanh saturates, and its derivative is at most 1 and usually
    far less.  Multiply L of those together, as the chain rule does when you
    backpropagate through a plain stack, and the gradient reaching the first
    layer decays roughly geometrically in L -- the vanishing-gradient problem.
    The skip adds a path whose local derivative is exactly 1, so d h_out/d h_in
    = 1 + (small), and gradients reach the bottom of the network intact.
    """
    h = params["W0"] @ feature(r, G)                       # Eq. (8)
    for W, b in params["layers"]:
        h = h + jnp.tanh(W @ h + b)                        # Eq. (9)
    return params["w0"] @ h + 1j * (params["w1"] @ h)      # Eq. (10)


# Same function, same parameters, a whole batch of positions at once.
# in_axes=(None, 0, None) reads: params not batched, r batched along axis 0,
# G not batched.  vmap rewrites the function to operate on the batch rather
# than looping in Python, so it stays a single fused GPU kernel.
orbital_batch = jax.vmap(orbital, in_axes=(None, 0, None))


def density(params, r, G):
    """|phi(r)|^2.  Real-valued, hence differentiable by jax.grad.

    jax.grad insists on a real scalar output, which is not JAX being fussy --
    a complex function has no single "gradient" unless you say what you mean
    (Wirtinger derivatives, or treating C as R^2).  |phi|^2 is a real
    observable, so its gradient is unambiguous.
    """
    phi = orbital(params, r, G)
    return jnp.real(phi * jnp.conj(phi))


if __name__ == "__main__":
    G = reciprocal(L_9SITE)
    params = init_params(jax.random.key(0))

    r = jnp.array([0.37, 1.10])
    print("phi(r)                =", orbital(params, r, G))

    rs = jnp.array([[0.0, 0.0], [0.37, 1.10], [1.5, 0.9]])
    print("orbital_batch(rs).shape =", orbital_batch(params, rs, G).shape)

    # The gradient of the density with respect to POSITION.  The same
    # machinery, with argnums=0, gives the gradient with respect to the
    # PARAMETERS -- which is what the VMC optimiser actually consumes.
    print("grad_r |phi|^2        =", jax.grad(density, argnums=1)(params, r, G))

    n_par = sum(x.size for x in jax.tree.leaves(params))
    print("n_parameters          =", n_par)
    # Cross-check against the paper's Appendix C formula with the attention
    # terms dropped and N_el = N_det = 1:
    #   2*d_dim*d_L + L*(d_L*(d_L+1)) + 2*d_L*N_el*N_det
    d_L, L_layers = 64, 3
    expected = 2 * 2 * d_L + L_layers * (d_L * (d_L + 1)) + 2 * d_L * 1 * 1
    print("Appendix C formula    =", expected, "match:", n_par == expected)
