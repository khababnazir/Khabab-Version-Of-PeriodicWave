"""The determinant step ALONE: time and memory of its Laplacian, four ways (Stage 8 speed study).

A made-up orbital-matrix function stands in for the network, so that ONLY the
way the determinants are added and differentiated changes between routes:

  folx_det         folx forward Laplacian + Stage 3's sum_of_determinants (plain det)
  folx_slogdet     folx forward Laplacian + determinants added from slogdet (the authors' way)
  hessian_det      Stage 4's full-Hessian kinetic energy + plain det
  hessian_slogdet  Stage 4's full-Hessian kinetic energy + slogdet

Usage:  python det_alone.py N_ELECTRONS ROUTE BATCH      (runs on GPU 0 only)
"""
import json
import os
import sys
import time

sys.path.insert(0, "/home/khabab/moire-nnvmc/learn/stage08_own_code")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_moire as tm
lock = tm.hold_lock(tm.GPU_LOCKS / "gpu0.lock", "GPU 0 is in use")      # noqa: F841
import jax
import jax.numpy as jnp
assert jax.default_backend() == "gpu"
jax.config.update("jax_default_matmul_precision", "float32")
import slaternet
from local_energy import kinetic_energy
from folx_kinetic import kinetic_energy_folx

n_el, route, batch = int(sys.argv[1]), sys.argv[2], int(sys.argv[3])
N_DET = 4
k1, k2, k3 = jax.random.split(jax.random.PRNGKey(0), 3)
WAVE = 0.3 * jax.random.normal(k1, (N_DET, n_el, n_el, 2 * n_el))   # every entry depends on every coordinate
AMP = jax.random.normal(k2, (N_DET, n_el, n_el)) + 1j * jax.random.normal(k3, (N_DET, n_el, n_el))


def matrices(positions):
    return AMP * jnp.exp(1j * (WAVE @ positions.reshape(-1)))


def sum_of_determinants_slogdet(m):
    sign, logdet = jnp.linalg.slogdet(m)
    shift = jax.lax.stop_gradient(jnp.max(logdet))
    total = jnp.sum(sign * jnp.exp(logdet - shift))
    return slaternet.LogPsi(log_abs=shift + jnp.log(jnp.abs(total)), phase=jnp.angle(total))


add_determinants = slaternet.sum_of_determinants if route.endswith("_det") else sum_of_determinants_slogdet
kinetic = kinetic_energy_folx if route.startswith("folx") else kinetic_energy
fn = jax.jit(jax.vmap(lambda R: kinetic(lambda X: add_determinants(matrices(X)), R)))
walkers = 3.0 * jax.random.uniform(jax.random.PRNGKey(3), (batch, n_el, 2))

result = dict(n_electrons=n_el, route=route, batch=batch)
try:
    compiled = fn.lower(walkers).compile()
    result["compiled_scratch_GiB"] = round(compiled.memory_analysis().temp_size_in_bytes / 2 ** 30, 4)
    times = []
    for i in range(6):
        start = time.time()
        out = jax.block_until_ready(compiled(walkers))
        times.append(time.time() - start)
    result["seconds_per_call"] = round(sum(times[1:]) / 5, 5)
    result["peak_GiB"] = round(jax.devices()[0].memory_stats().get("peak_bytes_in_use", 0) / 2 ** 30, 4)
    result["mean_kinetic_real"] = float(jnp.mean(out.real))
    result["mean_kinetic_imag"] = float(jnp.mean(out.imag))
except Exception as error:                       # report the failure, do not hide it
    result["error"] = type(error).__name__ + ": " + str(error).strip().splitlines()[0][:400]
print(json.dumps(result), flush=True)
