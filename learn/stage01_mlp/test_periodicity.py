"""
Stage 1 checks.  Run with:  python test_periodicity.py

No pytest -- it is not in the pinned environment, and these are plain asserts
so there is nothing to install.

The point of this file is not that the tests pass.  It is that they are
capable of failing.  A test suite that only ever asserts "these two numbers
are equal" cannot distinguish a correctly periodic network from a network
that ignores its input and returns a constant -- both pass.  So two of the
four checks below are negative controls: they assert that something DOES
change when it should.
"""
import jax

# Must come before any JAX operation.  JAX defaults to float32, in which
# sin(G.r) for |G.r| ~ 10 carries ~1e-7 relative error and a periodicity
# check can only be trusted to ~1e-5.  float64 lets us demand 1e-12 and so
# actually test the mathematics rather than the rounding.
#
# Production runs deliberately stay in float32 -- PeriodicWave even forces
# jax_default_matmul_precision to float32 for GPU throughput.  Knowing which
# regime you are in is the difference between a real failure and a yawn.
jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp

from orbital_forward import (L_9SITE, reciprocal, init_params, orbital,
                             orbital_batch, feature)

TOL = 1e-12
G = reciprocal(L_9SITE)
PARAMS = init_params(jax.random.key(7))
RNG = np.random.default_rng(0)


def check(name, condition, detail=""):
    print(f"{'PASS' if condition else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    assert condition, name


def worst_error(errors) -> float:
    """Largest error in a list -- and a guaranteed failure if any entry is NaN or inf.

    Why not max()?  Python's max(0.0, nan) returns 0.0, so accumulating with
    max() silently DROPS a NaN, and a broken function could print PASS.
    Returning +inf makes any "< TOL" comparison fail.
    """
    errors = np.asarray(errors, dtype=float)
    return float(np.max(errors)) if np.all(np.isfinite(errors)) else np.inf


def test_harness_rejects_nan():
    """NEGATIVE CONTROL on the test code itself: a NaN must never count as a pass."""
    check("worst_error() fails on NaN", not (worst_error([0.0, float("nan")]) < TOL))


def test_reciprocal_identity():
    """G_i . L_j = 2 pi delta_ij -- the identity everything else rests on."""
    err = np.abs(G @ L_9SITE - 2 * np.pi * np.eye(2)).max()
    check("G . L = 2*pi*I", err < TOL, f"max err {err:.2e}")


def test_periodic_under_lattice_vectors():
    """phi is unchanged by any integer combination of supercell vectors."""
    errors = []
    for _ in range(20):
        r = jnp.array(RNG.uniform(-3, 3, size=2))
        n, m = RNG.integers(-3, 4, size=2)
        shift = n * L_9SITE[:, 0] + m * L_9SITE[:, 1]
        d = abs(complex(orbital(PARAMS, r, G)) - complex(orbital(PARAMS, r + shift, G)))
        errors.append(d)
    worst = worst_error(errors)
    check("phi(r) == phi(r + n L1 + m L2)", worst < TOL, f"worst |delta| {worst:.2e}")


def test_not_constant():
    """NEGATIVE CONTROL 1: the network must actually depend on r.

    Without this, a network returning 0.0 everywhere would pass the
    periodicity test perfectly.
    """
    rs = jnp.array(RNG.uniform(-2, 2, size=(32, 2)))
    spread = float(jnp.std(jnp.abs(orbital_batch(PARAMS, rs, G))))
    check("phi varies with r", spread > 1e-6, f"std |phi| {spread:.3e}")


def test_non_lattice_shift_changes_phi():
    """NEGATIVE CONTROL 2: a shift that is NOT a lattice vector must matter.

    This is what separates "periodic with the right period" from "periodic
    with some other period", or from a feature map that accidentally collapsed
    the two directions together.
    """
    r = jnp.array([0.37, 1.10])
    bad = 0.5 * L_9SITE[:, 0] + 0.25 * L_9SITE[:, 1]   # fractional, not a lattice vector
    d = abs(complex(orbital(PARAMS, r, G)) - complex(orbital(PARAMS, r + bad, G)))
    check("phi(r) != phi(r + non-lattice shift)", d > 1e-6, f"|delta| {d:.3e}")


def test_feature_map_shape_and_range():
    f = feature(jnp.array([0.37, 1.10]), G)
    check("feature(r) has shape (4,)", f.shape == (4,))
    check("feature components in [-1, 1]", bool(jnp.all(jnp.abs(f) <= 1.0 + TOL)))


if __name__ == "__main__":
    print(f"jax x64 enabled: {jax.config.jax_enable_x64}\n")
    test_harness_rejects_nan()
    test_reciprocal_identity()
    test_periodic_under_lattice_vectors()
    test_not_constant()
    test_non_lattice_shift_changes_phi()
    test_feature_map_shape_and_range()
    print("\nall checks passed")
