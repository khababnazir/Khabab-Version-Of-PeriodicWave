"""
Stage 6, part 1 -- the energy gradient and the natural-gradient step.

Paper:   Sec. IV B, Eqs. (29)-(31)
Authors: loss.py (gradient), kfac_jax + curvature_tags_and_blocks.py (natural gradient)

1. THE GRADIENT.  Let O_k = d(log Psi)/d(theta_k), which is complex:
       O_k = d log|Psi| / d theta_k  +  i d(phase) / d theta_k.
   Then, averaging over configurations drawn from |Psi|^2,
       dE/d theta_k = 2 Re < (E_loc - E) * conj(O_k) >
   and splitting into real and imaginary parts,
       = 2 < (Re E_loc - E) * d log|Psi|/d theta_k >  +  2 < Im E_loc * d phase/d theta_k >.
   The second term needs Im E_loc.  The authors' loss.py formula contains it,
   but their fast kinetic-energy path (folx) passes in only Re E_loc, so the
   term is lost.  check_phase_gradient.py measures what that costs.

2. THE NATURAL GRADIENT.  Ordinary gradient descent measures the step length
   in parameter space, where some directions hardly change Psi and others
   change it a lot.  The natural gradient measures it in the space of wave
   functions instead, using the quantum geometric tensor (Eq. 30):
       S_kl = Re [ <conj(O_k) O_l> - <conj(O_k)> <O_l> ]
   and steps  d theta = - lr * (S + damping)^-1 * gradient   (Eq. 31).

   Here we build S exactly and solve that linear system; in the VMC
   literature this is called "stochastic reconfiguration".  S has P x P
   entries for P parameters, so this only works for small networks: the
   Stage 3 SlaterNet has 13,504 parameters, i.e. 182 million entries.
   KFAC, which the authors use, approximates S layer by layer as a Kronecker
   product of two small matrices and never builds S at all.  The authors'
   KFAC also keeps only the magnitude part of O (the Fisher information),
   as the paper states in Sec. IV B.
"""
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree


def log_derivatives(log_psi_fn, params, walkers):
    """O = d(log Psi)/d theta for every walker: complex array (n_walkers, n_parameters).

    log_psi_fn(params, positions) must return LogPsi(log_abs, phase).
    The parameters are flattened into one long vector, so that O is a plain matrix.
    """
    flat_params, unflatten = ravel_pytree(params)

    def log_abs(flat, positions):
        return log_psi_fn(unflatten(flat), positions).log_abs

    def phase(flat, positions):
        return log_psi_fn(unflatten(flat), positions).phase

    # jax.grad differentiates one walker; jax.vmap repeats it for every walker.
    d_log_abs = jax.vmap(jax.grad(log_abs), in_axes=(None, 0))(flat_params, walkers)
    d_phase = jax.vmap(jax.grad(phase), in_axes=(None, 0))(flat_params, walkers)
    return d_log_abs + 1j * d_phase


def energy_gradient(e_loc, O, weights=None):
    """dE/d theta = 2 Re < (E_loc - E) conj(O) >.

    e_loc:   (n_walkers,) complex local energies
    O:       (n_walkers, n_parameters) complex log-derivatives
    weights: how much each row counts.  None means equally, which is right for
             Monte Carlo samples (they are already distributed as |Psi|^2).
             check_phase_gradient.py passes exact grid weights instead.

    Pass a REAL e_loc to reproduce the authors' folx path: the Im E_loc term
    then drops out.
    """
    if weights is None:
        weights = jnp.full(e_loc.shape, 1.0 / e_loc.shape[0])

    energy = jnp.sum(weights * jnp.real(e_loc))
    deviation = e_loc - energy                               # imaginary part kept, if any
    return 2.0 * jnp.real(jnp.sum(weights[:, None] * deviation[:, None] * jnp.conj(O), axis=0))


def quantum_geometric_tensor(O):
    """S_kl = Re [ <conj(O_k) O_l> - <conj(O_k)><O_l> ],  Eq. (30), from samples."""
    centered = O - jnp.mean(O, axis=0)
    return jnp.real(centered.conj().T @ centered) / O.shape[0]


def clip_local_energy(e_loc, scale=5.0):
    """The authors' clipping (loss.py clip_local_values, their default settings).

    Rare configurations near a node of Psi can have enormous local energies,
    which would make single gradient steps wild.  For the GRADIENT only, pull
    each local energy back to within  scale x (mean absolute deviation)  of
    the batch mean.  Real and imaginary parts are clipped separately.
    The reported energy always uses the unclipped values.
    """
    def clip(values):
        center = jnp.mean(values)
        spread = jnp.mean(jnp.abs(values - center))
        return jnp.clip(values, center - scale * spread, center + scale * spread)

    return clip(jnp.real(e_loc)) + 1j * clip(jnp.imag(e_loc))


def natural_gradient_step(gradient, S, learning_rate, damping, max_step_norm):
    """Eq. (31):  step = - lr * (S + damping)^-1 gradient,  with a size limit.

    Why diagonalise S instead of just solving the linear system?  In exact
    arithmetic S is positive semi-definite -- it is an average of squares --
    but in floating point it shows small negative eigenvalues.  Measured in
    Stage 6: as low as -0.9 at the first training step in float32, larger than
    the damping.  Then S + damping is not positive, and the "step" can point
    uphill or blow up.  Those negative eigenvalues are pure rounding error, so
    we set them to zero before inverting.  S + damping is then strictly
    positive and the step is always downhill.

    The size limit caps how far Psi itself moves in one step, measured with
    S:  (step . S . step) <= max_step_norm.  This plays the role of the
    norm_constraint option in the authors' KFAC settings.
    """
    eigenvalues, eigenvectors = jnp.linalg.eigh(S)
    eigenvalues = jnp.maximum(eigenvalues, 0.0)                  # remove rounding error

    # Invert (S + damping) in its eigenbasis: divide each component by (eigenvalue + damping).
    gradient_components = eigenvectors.T @ gradient
    direction = eigenvectors @ (gradient_components / (eigenvalues + damping))
    step = -learning_rate * direction

    # step . S . step, using the cleaned S.  Never negative.
    step_components = eigenvectors.T @ step
    size = jnp.sum(eigenvalues * step_components**2)

    too_big = size > max_step_norm
    scale = jnp.where(too_big, jnp.sqrt(max_step_norm / jnp.maximum(size, 1e-30)), 1.0)
    return scale * step
