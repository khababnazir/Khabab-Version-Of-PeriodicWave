"""
Stage 5 -- the self-attention wave function, the paper's main ansatz.

Paper:   Sec. III C, Eqs. (15)-(23); Table II for the default sizes
Authors: external/PeriodicWave/periodicwave/CustomPsiformer.py

What changes from Stage 3.  In SlaterNet every electron went through the
network on its own, so electron i's orbital could not depend on where
electron j was.  That is the Hartree-Fock approximation.  Here, before each
perceptron layer, every electron's hidden state is updated with information
from all the other electrons: self-attention.  The orbitals become
"correlated orbitals" phi_j(r_i; all the other electrons), and their
determinant can describe correlation beyond Hartree-Fock.

    positions -> features -> embed -> [ attention -> perceptron ] x L -> orbitals -> det
                                        ^^^^^^^^^ the only new piece

The periodic features, the orbital projection and the determinant sum are
reused unchanged from Stage 3.

Paper vs authors' code.  Eq. (18) and the authors' jax.nn.dot_product_attention
look different, but describe the SAME family of wave functions -- they differ
only by constant rescalings that the learned weights absorb.  This file
follows the paper; test_psiformer.py proves the two agree exactly after the
rescaling.
"""
import sys
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "stage03_slater"))
from slaternet import (LogPsi, periodic_features, orbitals_from_hidden,  # noqa: E402
                       sum_of_determinants)


@dataclass(frozen=True)
class AttentionArchitecture:
    """Sizes of the network.  Defaults are the paper's Table II."""
    n_electrons: int
    n_determinants: int = 4
    n_layers: int = 3
    n_heads: int = 6
    attention_dim: int = 16     # d_Attn: size of keys and queries
    value_dim: int = 16         # d_AttnValues: not stated in the paper; the authors' config uses 16
    width: int = 64             # d_L: size of each electron's hidden state
    init_scale: float = 0.1


def init_parameters(key, arch: AttentionArchitecture) -> dict:
    """All variational parameters, as a nested dict.

    Per layer: one (W_k, W_q, W_v) triple per head, one output projection W_o,
    and one perceptron (W, b).  No biases in the attention, as in the paper's
    Eqs. (15)-(17) and the authors' code.
    """
    s = arch.init_scale
    width = arch.width

    key, key_embed, key_proj = jax.random.split(key, 3)
    params = {
        "embed": s * jax.random.normal(key_embed, (width, 4)),                   # W0, Eq. (8)
        "layers": [],
        "proj": s * jax.random.normal(                                           # Eq. (22)
            key_proj, (arch.n_determinants, arch.n_electrons, 2, width)),
    }

    for _ in range(arch.n_layers):
        heads = []
        for _ in range(arch.n_heads):
            key, key_k, key_q, key_v = jax.random.split(key, 4)
            heads.append({
                "W_k": s * jax.random.normal(key_k, (arch.attention_dim, width)),   # Eq. (15)
                "W_q": s * jax.random.normal(key_q, (arch.attention_dim, width)),   # Eq. (16)
                "W_v": s * jax.random.normal(key_v, (arch.value_dim, width)),       # Eq. (17)
            })

        key, key_o, key_w = jax.random.split(key, 3)
        params["layers"].append({
            "attention": {
                "heads": heads,
                "W_o": s * jax.random.normal(key_o, (width, arch.n_heads * arch.value_dim)),  # Eq. (19)
            },
            "perceptron": (s * jax.random.normal(key_w, (width, width)),             # Eq. (20)
                           jnp.zeros(width)),
        })

    return params


def self_attention(attention: dict, h: jax.Array) -> jax.Array:
    """Eqs. (15)-(19), one layer:  h (N, width) -> the attention update (N, width).

    For each head, with h_i the hidden state of electron i:
        key    k_i = W_k h_i     roughly, "what electron i is like"
        query  q_j = W_q h_j     roughly, "what electron j is looking for"
        value  v_j = W_v h_j     "what electron j passes on"
    Electron i then receives a weighted average of all the values v_j, with
    weight proportional to exp(q_j . k_i).  The heads are concatenated and
    projected back to the width by W_o.
    """
    head_outputs = []

    for head in attention["heads"]:
        keys = h @ head["W_k"].T          # (N, attention_dim): row i is k_i
        queries = h @ head["W_q"].T       # (N, attention_dim): row j is q_j
        values = h @ head["W_v"].T        # (N, value_dim):     row j is v_j

        # scores[i, j] = k_i . q_j  -- Eq. (18) compares query j with key i.
        scores = keys @ queries.T         # (N, N)

        # Weights: exp(score) normalised over j, i.e. a softmax of each row.
        # jax.nn.softmax subtracts each row's maximum before exp(), so it
        # cannot overflow; the result is exactly exp(s_ij) / sum_j exp(s_ij).
        weights = jax.nn.softmax(scores, axis=1)       # every row sums to 1

        # Weighted average of the values, then Eq. (18)'s factor 1/sqrt(d_values).
        value_dim = values.shape[1]
        head_outputs.append(weights @ values / jnp.sqrt(value_dim))   # (N, value_dim)

    all_heads = jnp.concatenate(head_outputs, axis=1)                 # (N, n_heads * value_dim)
    return all_heads @ attention["W_o"].T                              # (N, width)


def hidden_states(params: dict, positions: jax.Array, G: jax.Array) -> jax.Array:
    """(N, 2) positions -> (N, width) hidden states, Eqs. (8), (19), (20).

    Each layer has two residual steps: attention, then a perceptron.
    Compare Stage 3, where each layer was the perceptron alone.
    """
    h = periodic_features(positions, G) @ params["embed"].T           # Eq. (8)

    for layer in params["layers"]:
        f = h + self_attention(layer["attention"], h)                 # Eq. (19)
        weight, bias = layer["perceptron"]
        h = f + jnp.tanh(f @ weight.T + bias)                         # Eq. (20)

    return h


def log_psi(params: dict, positions: jax.Array, G: jax.Array) -> LogPsi:
    """Eq. (23) for the attention network: log|Psi| and arg Psi."""
    h = hidden_states(params, positions, G)
    matrices = orbitals_from_hidden(params["proj"], h)
    return sum_of_determinants(matrices)


def appendix_c_parameter_count(arch: AttentionArchitecture) -> int:
    """The paper's Appendix C formula for the number of parameters.

        N_par = 2 d_dim d_L                                    embedding
              + L ( 2 N_heads d_Attn d_L                       keys and queries
                  + 2 N_heads d_AttnValues d_L                 values and W_o
                  + d_L (d_L + 1) )                            perceptron W and b
              + 2 d_L N_el N_det                               orbital projections
    """
    d_dim = 2
    per_layer = (2 * arch.n_heads * arch.attention_dim * arch.width
                 + 2 * arch.n_heads * arch.value_dim * arch.width
                 + arch.width * (arch.width + 1))
    return (2 * d_dim * arch.width
            + arch.n_layers * per_layer
            + 2 * arch.width * arch.n_electrons * arch.n_determinants)


if __name__ == "__main__":
    import time
    import numpy as np
    from slaternet import (Architecture, supercell_vectors, reciprocal_vectors,
                           init_parameters as init_slaternet, hidden_states as slaternet_hidden)

    lattice = supercell_vectors(9)
    G = reciprocal_vectors(lattice)
    arch = AttentionArchitecture(n_electrons=6)
    params = init_parameters(jax.random.key(0), arch)

    n_par = sum(x.size for x in jax.tree.leaves(params))
    print(f"parameters: {n_par}   Appendix C formula: {appendix_c_parameter_count(arch)}")
    table1 = AttentionArchitecture(n_electrons=18)
    print(f"at Table I size (18 electrons): {appendix_c_parameter_count(table1)}")

    positions = jnp.asarray(np.random.default_rng(0).uniform(0, 1, (6, 2)) @ lattice.T)
    out = log_psi(params, positions, G)
    print(f"log|Psi| = {float(out.log_abs):+.6f}   phase = {float(out.phase):+.6f}")

    # The whole point of attention: move electron 5 and watch electron 0's hidden state.
    moved = positions.at[5].add(jnp.array([0.3, 0.1]))
    change_attention = float(jnp.max(jnp.abs(hidden_states(params, moved, G)[0]
                                             - hidden_states(params, positions, G)[0])))
    sn_params = init_slaternet(jax.random.key(0), Architecture(n_electrons=6))
    change_slaternet = float(jnp.max(jnp.abs(slaternet_hidden(sn_params, moved, G)[0]
                                             - slaternet_hidden(sn_params, positions, G)[0])))
    print("move electron 5; largest change in electron 0's hidden state:")
    print(f"   SlaterNet: {change_slaternet:.2e}     attention network: {change_attention:.2e}")

    batch = jax.jit(jax.vmap(lambda R: log_psi(params, R, G).log_abs))
    walkers = jax.random.uniform(jax.random.key(1), (1024, 6, 2)) @ jnp.asarray(lattice).T
    batch(walkers).block_until_ready()
    t0 = time.perf_counter()
    for _ in range(10):
        batch(walkers).block_until_ready()
    print(f"1024 walkers, compiled, CPU: {(time.perf_counter() - t0) / 10 * 1e3:.1f} ms per call")
