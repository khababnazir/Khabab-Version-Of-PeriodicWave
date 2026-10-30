"""What happens to each way of adding determinants when ONE determinant is exactly zero (Stage 8 speed study).

Psi(e) = det A(e) + det B,  A(e) = [[1, 1], [1, 1 + e]] (det A = e, singular at e = 0),  B = diag(2, 1) (det B = 2).
Exact:  Psi = e + 2,   d log|Psi| / de = 1 / (e + 2) = 0.5 at e = 0.     Runs on the CPU.
"""
import json
import sys
import jax
import jax.numpy as jnp
sys.path.insert(0, "/home/khabab/moire-nnvmc/learn/stage03_slater")
import slaternet

precision = sys.argv[1]
jax.config.update("jax_enable_x64", precision == "float64")
dtype = jnp.float64 if precision == "float64" else jnp.float32


def sum_of_determinants_slogdet(m):
    sign, logdet = jnp.linalg.slogdet(m)
    shift = jax.lax.stop_gradient(jnp.max(logdet))
    total = jnp.sum(sign * jnp.exp(logdet - shift))
    return slaternet.LogPsi(log_abs=shift + jnp.log(jnp.abs(total)), phase=jnp.angle(total))


def matrices(e):
    A = jnp.array([[1.0, 1.0], [1.0, 1.0]], dtype) + e * jnp.array([[0.0, 0.0], [0.0, 1.0]], dtype)
    B = jnp.array([[2.0, 0.0], [0.0, 1.0]], dtype)
    return jnp.stack([A, B])


for e in [1e-1, 1e-3, 1e-5, 1e-7, 0.0]:
    e = jnp.asarray(e, dtype)
    row = dict(precision=precision, e=float(e), exact=1.0 / (float(e) + 2.0))
    for name, add in [("det", slaternet.sum_of_determinants), ("slogdet", sum_of_determinants_slogdet)]:
        f = lambda e: add(matrices(e)).log_abs                     # noqa: E731
        row[name] = float(jax.grad(f)(e))
        row[name + "_second"] = float(jax.grad(jax.grad(f))(e))
    row["exact_second"] = -1.0 / (float(e) + 2.0) ** 2
    print(json.dumps(row))
