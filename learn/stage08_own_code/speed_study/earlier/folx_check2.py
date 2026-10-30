"""folx kinetic energy == Stage 4, 18 electrons, double precision, CPU; attention made exactly double precision."""
import os, sys
os.environ["JAX_PLATFORMS"] = "cpu"
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
sys.path.insert(0, "/home/khabab/moire-nnvmc/learn/stage08_own_code"); sys.path.insert(0, "/tmp")
import train_moire  # noqa: F401
import slaternet, psiformer
from local_energy import kinetic_energy
import folx_kinetic

library_attention = jax.nn.dot_product_attention
def exact_attention(q, k, v, *args, **kwargs):      # (N, heads, d): softmax written out, in the input precision
    assert not args and not kwargs
    w = jax.nn.softmax(jnp.einsum("tnh,snh->nts", q, k) / jnp.sqrt(q.shape[-1]), axis=-1)
    return jnp.einsum("nts,snh->tnh", w, v)
q, k, v = (jax.random.normal(jax.random.PRNGKey(i), (18, 4, 16), dtype=jnp.float32) for i in range(3))
print("exact vs library attention, float32 inputs: max |diff|", float(jnp.max(jnp.abs(exact_attention(q, k, v) - library_attention(q, k, v)))))
jax.nn.dot_product_attention = exact_attention

lattice = slaternet.supercell_vectors(27); G = slaternet.reciprocal_vectors(lattice)
R = jax.random.uniform(jax.random.PRNGKey(5), (3, 18, 2), dtype=jnp.float64) @ jnp.asarray(lattice).T
for name, arch, init, lp in [("slaternet", slaternet.ReleasedArchitecture(n_electrons=18), slaternet.init_released, slaternet.log_psi_released),
                             ("attention", psiformer.ReleasedAttentionArchitecture(n_electrons=18), psiformer.init_released, psiformer.log_psi_released)]:
    params = init(jax.random.PRNGKey(1), arch)
    psi = lambda X: lp(params, X, G, arch)
    ours = jnp.stack([jax.jit(lambda r: kinetic_energy(psi, r))(r) for r in R])
    for sparsity in (0.0, 0.6):
        folx_kinetic.SPARSITY = sparsity
        new = jnp.stack([jax.jit(lambda r: folx_kinetic.kinetic_energy_folx(psi, r))(r) for r in R])
        print(f"{name:9s} sparsity {sparsity}: max |difference| / max |value|  real "
              f"{float(jnp.max(jnp.abs(ours.real - new.real)) / jnp.max(jnp.abs(ours.real))):.1e}  imag "
              f"{float(jnp.max(jnp.abs(ours.imag - new.imag)) / jnp.max(jnp.abs(ours.imag))):.1e}", flush=True)
