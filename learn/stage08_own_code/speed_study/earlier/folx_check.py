"""folx kinetic energy == Stage 4 kinetic energy, 18 electrons, 27 sites, double precision, CPU."""
import os, sys
os.environ["JAX_PLATFORMS"] = "cpu"
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
sys.path.insert(0, "/home/khabab/moire-nnvmc/learn/stage08_own_code"); sys.path.insert(0, "/tmp")
import train_moire  # noqa: F401  (puts Stages 3-7 on the path)
import slaternet, psiformer
from local_energy import kinetic_energy
from folx_kinetic import kinetic_energy_folx
lattice = slaternet.supercell_vectors(27); G = slaternet.reciprocal_vectors(lattice)
R = jax.random.uniform(jax.random.PRNGKey(5), (3, 18, 2), dtype=jnp.float64) @ jnp.asarray(lattice).T
for name, arch, init, lp in [("slaternet", slaternet.ReleasedArchitecture(n_electrons=18), slaternet.init_released, slaternet.log_psi_released),
                             ("attention", psiformer.ReleasedAttentionArchitecture(n_electrons=18), psiformer.init_released, psiformer.log_psi_released)]:
    params = init(jax.random.PRNGKey(1), arch)
    psi = lambda X: lp(params, X, G, arch)
    ours = jnp.stack([jax.jit(lambda r: kinetic_energy(psi, r))(r) for r in R])
    new = jnp.stack([jax.jit(lambda r: kinetic_energy_folx(psi, r))(r) for r in R])
    print(f"{name}: Stage 4 {np.round(np.asarray(ours), 6)}")
    print(f"{name}: folx    {np.round(np.asarray(new), 6)}")
    print(f"{name}: max |difference| / max |value|  real {float(jnp.max(jnp.abs(ours.real - new.real)) / jnp.max(jnp.abs(ours.real))):.1e}  "
          f"imag {float(jnp.max(jnp.abs(ours.imag - new.imag)) / jnp.max(jnp.abs(ours.imag))):.1e}", flush=True)
