"""
Stage 5 checks.  Run:  python test_psiformer.py

  1. attention is permutation equivariant           (why antisymmetry survives)
  2. paper's Eq. (18) == the authors' attention      (after rescaling weights)
  3. the wave function is antisymmetric
  4. the wave function is periodic in every electron
  5. electrons actually see each other               (and in SlaterNet they do not)
  6. parameter count == the paper's Appendix C formula
"""
import sys
from pathlib import Path

from absl.testing import absltest

import jax
jax.config.update("jax_enable_x64", True)     # before any JAX operation
import numpy as np
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "stage03_slater"))
import slaternet  # noqa: E402
from psiformer import (AttentionArchitecture, init_parameters, self_attention,  # noqa: E402
                       hidden_states, log_psi, appendix_c_parameter_count)

N = 6
LATTICE = slaternet.supercell_vectors(9)
G = slaternet.reciprocal_vectors(LATTICE)
ARCH = AttentionArchitecture(n_electrons=N)
PARAMS = init_parameters(jax.random.key(0), ARCH)


def random_positions(seed):
    rng = np.random.default_rng(seed)
    return jnp.asarray(rng.uniform(0, 1, (N, 2)) @ LATTICE.T)


class AttentionLayerTest(absltest.TestCase):

    def test_is_permutation_equivariant(self):
        """Relabelling the electrons relabels the outputs, and changes nothing else.

        This is the property (paper Eq. 21) that keeps the determinant
        antisymmetric even though each orbital now depends on every electron.
        """
        attention = PARAMS["layers"][0]["attention"]
        h = jnp.asarray(np.random.default_rng(1).normal(size=(N, ARCH.width)))
        order = np.random.default_rng(2).permutation(N)

        np.testing.assert_allclose(self_attention(attention, h[order]),
                                   self_attention(attention, h)[order],
                                   rtol=1e-12, atol=1e-14)

    def test_equals_the_authors_attention_after_rescaling(self):
        """Eq. (18) and jax.nn.dot_product_attention are the same function family.

        jax.nn.dot_product_attention computes, for each electron i,
            sum_j softmax_j( q'_i . k'_j / sqrt(d_attn) ) v'_j
        Choosing  q' from sqrt(d_attn) * W_k,  k' from W_q,  and W_o / sqrt(d_values)
        turns that into exactly Eq. (18).  So the paper and the code differ
        only in how the same weights are scaled, not in what they can represent.
        """
        attention = PARAMS["layers"][0]["attention"]
        h = jnp.asarray(np.random.default_rng(3).normal(size=(N, ARCH.width)))
        d_attn, d_values = ARCH.attention_dim, ARCH.value_dim

        q_authors = jnp.stack([h @ (np.sqrt(d_attn) * head["W_k"]).T for head in attention["heads"]], axis=1)
        k_authors = jnp.stack([h @ head["W_q"].T for head in attention["heads"]], axis=1)
        v_authors = jnp.stack([h @ head["W_v"].T for head in attention["heads"]], axis=1)
        # shapes (N, n_heads, dim) -- the layout the authors pass in CustomPsiformer.py

        out = jax.nn.dot_product_attention(q_authors, k_authors, v_authors)   # (N, n_heads, d_values)
        out = out.reshape(N, -1) @ (attention["W_o"] / np.sqrt(d_values)).T

        # Tolerance is single-precision level on purpose.  Measured in JAX 0.8.0:
        # jax.nn.dot_product_attention returns float64 arrays for float64 inputs
        # but is only about 1e-7 accurate against an explicit float64 softmax,
        # so it works at roughly float32 accuracy internally.  A wrong mapping
        # (a missing 1/sqrt(d), keys and queries the wrong way round) is off by
        # order 1, so this tolerance still separates right from wrong cleanly.
        np.testing.assert_allclose(out, self_attention(attention, h), rtol=1e-5, atol=1e-7)


class WaveFunctionTest(absltest.TestCase):

    def test_antisymmetric_under_exchange(self):
        """Swap two electrons: |Psi| unchanged, Psi picks up a factor -1."""
        positions = random_positions(4)
        swapped = positions.at[jnp.array([1, 4])].set(positions[jnp.array([4, 1])])
        a = log_psi(PARAMS, positions, G)
        b = log_psi(PARAMS, swapped, G)

        np.testing.assert_allclose(b.log_abs, a.log_abs, rtol=1e-10)
        np.testing.assert_allclose(np.exp(1j * (b.phase - a.phase)), -1.0, atol=1e-10)

    def test_periodic_in_each_electron(self):
        """Move one electron by a supercell vector: Psi unchanged."""
        positions = random_positions(5)
        for electron in range(N):
            shifted = positions.at[electron].add(LATTICE[:, 0] - 2 * LATTICE[:, 1])
            a = log_psi(PARAMS, positions, G)
            b = log_psi(PARAMS, shifted, G)
            np.testing.assert_allclose(b.log_abs, a.log_abs, rtol=1e-10)
            np.testing.assert_allclose(np.exp(1j * (b.phase - a.phase)), 1.0, atol=1e-10)

    def test_electrons_see_each_other(self):
        """Moving electron 5 must change electron 0's hidden state -- and in
        SlaterNet it must not.  This is the whole difference between the two.
        """
        positions = random_positions(6)
        moved = positions.at[5].add(jnp.array([0.3, 0.1]))

        change = jnp.max(jnp.abs(hidden_states(PARAMS, moved, G)[0]
                                 - hidden_states(PARAMS, positions, G)[0]))
        self.assertGreater(float(change), 1e-4)

        sn_params = slaternet.init_parameters(jax.random.key(0), slaternet.Architecture(n_electrons=N))
        sn_change = jnp.max(jnp.abs(slaternet.hidden_states(sn_params, moved, G)[0]
                                    - slaternet.hidden_states(sn_params, positions, G)[0]))
        np.testing.assert_allclose(sn_change, 0.0, atol=1e-14)

    def test_parameter_count_matches_appendix_c(self):
        n_par = sum(x.size for x in jax.tree.leaves(PARAMS))
        self.assertEqual(n_par, appendix_c_parameter_count(ARCH))
        self.assertEqual(n_par, 89536)       # Table II sizes, 6 electrons


if __name__ == "__main__":
    absltest.main()
