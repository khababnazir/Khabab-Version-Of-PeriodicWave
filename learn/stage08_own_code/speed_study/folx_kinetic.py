"""Stage 4's kinetic energy, with the derivatives from folx's forward Laplacian (smoke test, not in the repo)."""
import jax
import jax.numpy as jnp
import os
import folx

SPARSITY = float(os.environ.get("FOLX_SPARSITY", "0"))   # folx sparsity_threshold (0 = dense)


def kinetic_energy_folx(log_psi_fn, positions):
    """-1/2 (Laplacian Psi) / Psi for one configuration: the formula of Stage 4's kinetic_energy.

    folx returns, in one forward pass, the value, the first derivatives and the
    Laplacian of [log|Psi|, phase] -- instead of two full 2N x 2N Hessians.
    """
    shape = positions.shape
    x = positions.reshape(-1)

    def u_and_phi(x):
        out = log_psi_fn(x.reshape(shape))
        return jnp.stack([out.log_abs, out.phase])

    result = folx.forward_laplacian(u_and_phi, sparsity_threshold=SPARSITY)(x)
    laplacian_u, laplacian_phi = result.laplacian[0], result.laplacian[1]
    jacobian = result.jacobian.dense_array          # (2N, 2): d[log|Psi|, phase] / dx
    grad_u, grad_phi = jacobian[:, 0], jacobian[:, 1]
    real_part = laplacian_u + jnp.sum(grad_u ** 2) - jnp.sum(grad_phi ** 2)
    imag_part = laplacian_phi + 2.0 * jnp.sum(grad_u * grad_phi)
    return -0.5 * (real_part + 1j * imag_part)
