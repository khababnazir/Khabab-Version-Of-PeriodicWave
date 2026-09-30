"""
Stage 6, part 2 -- does dropping Im E_loc bias the energy gradient?

The open question since Stage 3.  The authors' fast path keeps only
Re E_loc; that removes the term  2 < Im E_loc * d(phase)/d theta >  from
the gradient.  Is that term actually nonzero?

The method: NO Monte Carlo at all.  For ONE electron in the supercell,
the "average over |Psi|^2" is a 2D integral, which a uniform grid does
essentially exactly (the integrand is smooth and periodic).  So we can
compute three gradients without any sampling noise:

  exact      d/d theta of the energy, the energy written as
             E = sum_grid p(r) [ |grad log Psi|^2 / 2 + V(r) ],   p = |Psi|^2 normalised,
             which uses only FIRST derivatives of Psi (integrate the
             kinetic energy by parts).  Independent of the local-energy code.
  full       the estimator  2 Re < (E_loc - E) conj(O) >  with complex E_loc
  real only  the same estimator with Re E_loc only (the authors' folx path)

and also the authors' own loss.py formula written out literally, fed with
complex E_loc (their default path) and with real E_loc (their folx path).

Two wave functions:
  complex  a SlaterNet orbital with both real and imaginary projections
  real     the same, with the imaginary projection set to zero: the phase
           is then 0 or pi, locally constant, and the lost term must vanish.

Run:  python check_phase_gradient.py
"""
import sys
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "stage03_slater"))
sys.path.insert(0, str(HERE.parent / "stage04_mcmc"))
from slaternet import Architecture, init_parameters, log_psi, supercell_vectors, reciprocal_vectors  # noqa: E402
from local_energy import moire_units, moire_potential, local_energy  # noqa: E402
from gradient import log_derivatives, energy_gradient  # noqa: E402

LATTICE = supercell_vectors(9)
G = reciprocal_vectors(LATTICE)
ENERGY_UNIT_MEV, STRENGTH = moire_units()
PHI = np.pi / 4


def grid_points(n):
    """n x n uniform grid over the supercell, as (n*n, 1, 2): one electron per point."""
    fractions = (np.arange(n) + 0.5) / n
    f1, f2 = np.meshgrid(fractions, fractions, indexing="ij")
    fractional = np.stack([f1.ravel(), f2.ravel()], axis=1)
    return jnp.asarray(fractional @ LATTICE.T)[:, None, :]


def wave_function(params):
    return lambda R: log_psi(params, R, G)


def exact_energy(params, points):
    """E = sum_r p(r) [ (|grad log|Psi||^2 + |grad phase|^2) / 2 + V(r) ].

    Uses |grad Psi|^2 = |Psi|^2 ( |grad log|Psi||^2 + |grad phase|^2 ), i.e.
    the kinetic energy integrated by parts -- no second derivatives, and no
    use of the local-energy code we are testing.
    """
    psi = wave_function(params)

    def log_abs(R):
        return psi(R).log_abs

    def phase(R):
        return psi(R).phase

    u = jax.vmap(log_abs)(points)
    grad_u = jax.vmap(jax.grad(log_abs))(points)
    grad_phase = jax.vmap(jax.grad(phase))(points)
    potential = jax.vmap(lambda R: moire_potential(R, STRENGTH, PHI))(points)

    p = jax.nn.softmax(2.0 * u)                   # |Psi|^2 / sum |Psi|^2, computed stably
    kinetic = 0.5 * (jnp.sum(grad_u**2, axis=(1, 2)) + jnp.sum(grad_phase**2, axis=(1, 2)))
    return jnp.sum(p * (kinetic + potential))


def estimators(params, points):
    """Gradient estimators, evaluated with exact grid weights instead of samples."""
    psi = wave_function(params)
    e_loc = jax.vmap(lambda R: local_energy(psi, R, STRENGTH, PHI))(points)
    O = log_derivatives(lambda p, R: log_psi(p, R, G), params, points)
    p = jax.nn.softmax(2.0 * jax.vmap(lambda R: psi(R).log_abs)(points))   # exact |Psi|^2 weights

    def authors_formula(e):
        """loss.py, complex branch, literally: (term1 - 2 term2), with no clipping.

        clipped_el = E_loc;  clipped_energy = its mean;  psi_tangent = O.
        """
        clipped_energy = jnp.sum(p * e)
        term1 = (jnp.sum(p[:, None] * e[:, None] * jnp.conj(O), axis=0)
                 + jnp.sum(p[:, None] * jnp.conj(e)[:, None] * O, axis=0))
        term2 = jnp.sum(p[:, None] * clipped_energy * jnp.real(O), axis=0)
        return jnp.real(term1 - 2.0 * term2)

    energy_from_e_loc = float(jnp.sum(p * jnp.real(e_loc)))
    return {
        "full (gradient.py)":         energy_gradient(e_loc, O, weights=p),
        "real only (gradient.py)":    energy_gradient(jnp.real(e_loc) + 0j, O, weights=p),
        "authors, complex E_loc":     authors_formula(e_loc),
        "authors, real E_loc (folx)": authors_formula(jnp.real(e_loc) + 0j),
    }, energy_from_e_loc


def report(label, params, n_grid):
    points = grid_points(n_grid)
    e_exact = float(exact_energy(params, points))
    g_exact, _ = ravel_pytree(jax.grad(exact_energy)(params, points))
    results, e_from_eloc = estimators(params, points)

    print(f"\n{label}   (grid {n_grid} x {n_grid})")
    print(f"  energy: exact {e_exact:+.10f}   from <Re E_loc> {e_from_eloc:+.10f}   "
          f"difference {abs(e_exact - e_from_eloc):.1e}")
    norm = float(jnp.linalg.norm(g_exact))
    print(f"  |exact gradient| = {norm:.6e}")
    for name, g in results.items():
        rel = float(jnp.linalg.norm(g - g_exact)) / norm
        print(f"  {name:28s} relative error vs exact: {rel:.2e}")
    return g_exact


if __name__ == "__main__":
    arch = Architecture(n_electrons=1, n_layers=2, width=32, init_scale=0.3)
    complex_params = init_parameters(jax.random.key(0), arch)
    real_params = dict(complex_params, proj=complex_params["proj"].at[:, :, 1, :].set(0.0))
    print(f"one electron, {sum(x.size for x in jax.tree.leaves(complex_params))} parameters")

    # Quadrature convergence first: the exact gradient must not change with the grid.
    g48 = report("complex wave function", complex_params, 48)
    g64 = report("complex wave function", complex_params, 64)
    print(f"\n  exact gradient, grid 48 vs 64: relative change "
          f"{float(jnp.linalg.norm(g48 - g64) / jnp.linalg.norm(g64)):.1e}")

    report("real wave function (imaginary projection zero)", real_params, 64)
