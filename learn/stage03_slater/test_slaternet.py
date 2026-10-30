"""
Stage 3 checks.  Run:  python test_slaternet.py

Antisymmetry, Pauli and periodicity must hold for EVERY parameter value --
they are properties of how the ansatz is written, not things training learns.
Each is checked for a 1-determinant (Hartree-Fock) and a 4-determinant net.

Rule for this file: every comparison goes through worst_error() or an explicit
finiteness check.  Never Python's max() or min(): max(1e-16, nan) == 1e-16,
so a NaN would silently count as a pass.
"""
import jax
jax.config.update("jax_enable_x64", True)          # before any JAX operation

import numpy as np
import jax.numpy as jnp

from slaternet import (Architecture, init_parameters, log_psi, orbital_matrices,
                       supercell_vectors, reciprocal_vectors, sum_of_determinants_slogdet)

TOL = 1e-10
N = 6
LATTICE = supercell_vectors(9)
G = reciprocal_vectors(LATTICE)
RNG = np.random.default_rng(0)

NETS = {
    "1 det": init_parameters(jax.random.key(11), Architecture(n_electrons=N, n_determinants=1)),
    "4 det": init_parameters(jax.random.key(12), Architecture(n_electrons=N, n_determinants=4)),
}


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def random_positions() -> jnp.ndarray:
    return jnp.asarray(RNG.uniform(0, 1, (N, 2)) @ LATTICE.T)


def wrap(angle: float) -> float:
    """Map an angle into [-pi, pi)."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    assert ok, name


def worst_error(errors) -> float:
    """Largest error in a list, or +inf if any entry is NaN or inf.

    +inf makes every "< TOL" comparison fail, so a broken value can never pass.
    """
    errors = np.asarray(errors, dtype=float)
    return float(np.max(errors)) if np.all(np.isfinite(errors)) else np.inf


def all_finite(tree) -> bool:
    return all(bool(jnp.all(jnp.isfinite(x))) for x in jax.tree.leaves(tree))


def as_complex(out) -> complex:
    return complex(np.exp(float(out.log_abs) + 1j * float(out.phase)))


def naive_log_abs(params, positions):
    """log|sum_m det A_m| the obvious way.

    Unsafe at 18 electrons (Psi underflows), fine at 6.  Its derivatives go
    through jnp.linalg.det directly, so it is our ground truth here.
    """
    return jnp.log(jnp.abs(jnp.sum(jnp.linalg.det(orbital_matrices(params, positions, G)))))


def singular_variants(params):
    """The 4-det net, plus two versions in which determinant 0 alone is singular."""
    proj = params["proj"]
    return {
        "ordinary":                        params,
        "det 0 has two equal orbitals":    dict(params, proj=proj.at[0, 1].set(proj[0, 0])),   # rank N-1
        "det 0 is exactly zero":           dict(params, proj=proj.at[0].set(0.0)),             # rank 0
    }


# ---------------------------------------------------------------------------
#  Checks
# ---------------------------------------------------------------------------

def test_harness_rejects_nan():
    """NEGATIVE CONTROL on the test code itself: a NaN must never count as a pass."""
    check("worst_error() fails on NaN", not (worst_error([0.0, float("nan")]) < TOL))


def test_antisymmetry():
    """Swapping two electrons: |Psi| unchanged, phase shifts by pi."""
    for label, params in NETS.items():
        abs_errs, phase_errs = [], []
        for _ in range(10):
            r = random_positions()
            i, j = RNG.choice(N, size=2, replace=False)
            swapped = r.at[jnp.array([i, j])].set(r[jnp.array([j, i])])
            a, b = log_psi(params, r, G), log_psi(params, swapped, G)
            abs_errs.append(abs(float(b.log_abs - a.log_abs)))
            phase_errs.append(abs(wrap(float(b.phase - a.phase) - np.pi)))
        worst_abs, worst_phase = worst_error(abs_errs), worst_error(phase_errs)
        check(f"[{label}] |Psi| unchanged under exchange", worst_abs < TOL, f"worst {worst_abs:.1e}")
        check(f"[{label}] phase flips by pi under exchange", worst_phase < TOL, f"worst {worst_phase:.1e}")


def test_pauli_exclusion():
    """Two electrons at the same point -> identical rows -> Psi = 0.

    Exactly -inf in exact arithmetic; LU rounding leaves a residue, so we
    demand that |Psi| collapses by many orders of magnitude.
    """
    for label, params in NETS.items():
        r = random_positions()
        normal = float(log_psi(params, r, G).log_abs)
        collided = float(log_psi(params, r.at[1].set(r[0]), G).log_abs)
        drop = (normal - collided) / np.log(10)
        check(f"[{label}] Psi -> 0 when two electrons coincide", drop > 10,
              f"|Psi| falls by 10^{drop:.0f}")


def test_periodicity_per_electron():
    """Moving one electron by a supercell vector changes nothing."""
    for label, params in NETS.items():
        errs = []
        for _ in range(10):
            r = random_positions()
            i = RNG.integers(N)
            n, m = RNG.integers(-2, 3, size=2)
            shifted = r.at[i].add(n * LATTICE[:, 0] + m * LATTICE[:, 1])
            a, b = log_psi(params, r, G), log_psi(params, shifted, G)
            errs += [abs(float(b.log_abs - a.log_abs)), abs(wrap(float(b.phase - a.phase)))]
        worst = worst_error(errs)
        check(f"[{label}] Psi unchanged when one electron moves by L", worst < TOL, f"worst {worst:.1e}")


def test_log_domain_matches_direct_sum():
    """The rescaled computation must equal the naive sum of determinants."""
    for label, params in singular_variants(NETS["4 det"]).items():
        r = random_positions()
        direct = complex(jnp.sum(jnp.linalg.det(orbital_matrices(params, r, G))))
        rel = abs(as_complex(log_psi(params, r, G)) - direct) / abs(direct)
        check(f"[4 det, {label}] log_psi == direct sum of determinants",
              rel < 1e-9, f"rel err {rel:.1e}")


def test_one_determinant_is_a_plain_slater_determinant():
    """With D = 1, log_psi must reduce to slogdet of the single orbital matrix."""
    params = NETS["1 det"]
    r = random_positions()
    sign, logdet = jnp.linalg.slogdet(orbital_matrices(params, r, G)[0])
    out = log_psi(params, r, G)
    err = worst_error([abs(float(out.log_abs - logdet)),
                       abs(wrap(float(out.phase - np.angle(sign))))])
    check("1 det: log_psi == slogdet of the single matrix", err < TOL, f"err {err:.1e}")


def test_log_psi_really_sums_the_determinants():
    """NEGATIVE CONTROL, through log_psi itself.

    The 4-determinant Psi must differ from Psi with each determinant kept
    alone.  Catches a slip inside log_psi that collapses the sum to one term.
    """
    params = NETS["4 det"]
    r = random_positions()
    psi_full = as_complex(log_psi(params, r, G))
    gaps = np.array([abs(psi_full - as_complex(log_psi(dict(params, proj=params["proj"][m:m + 1]), r, G)))
                     / abs(psi_full) for m in range(4)])
    ok = bool(np.all(np.isfinite(gaps))) and bool(np.all(gaps > 1e-3))
    check("4 det: Psi differs from every single determinant", ok,
          f"smallest gap {np.min(gaps):.2f}")


def test_exactly_zero_psi_gives_minus_inf_not_nan():
    """Every determinant exactly zero -> log|Psi| = -inf and phase = 0, never NaN."""
    for label, params in NETS.items():
        dead = dict(params, proj=jnp.zeros_like(params["proj"]))
        cases = {"zero projections": log_psi(dead, random_positions(), G),
                 "all electrons at one point": log_psi(params, jnp.zeros((N, 2)), G)}
        for how, out in cases.items():
            ok = bool(jnp.isneginf(out.log_abs)) and float(out.phase) == 0.0
            check(f"[{label}] {how}: log|Psi| = -inf, phase = 0", ok,
                  f"got log|Psi| = {float(out.log_abs)}, phase = {float(out.phase)}")


def test_parameter_gradients_match_naive_formula():
    """d log|Psi| / d theta: finite, and equal to the naive formula's --
    including when ONE determinant is singular and the others are not.
    """
    r = random_positions()
    for label, params in singular_variants(NETS["4 det"]).items():
        g_ours = jax.grad(lambda p: log_psi(p, r, G).log_abs)(params)
        g_naive = jax.grad(naive_log_abs)(params, r)
        scale = worst_error([float(jnp.max(jnp.abs(x))) for x in jax.tree.leaves(g_naive)])
        diff = worst_error([float(jnp.max(jnp.abs(a - b)))
                            for a, b in zip(jax.tree.leaves(g_ours), jax.tree.leaves(g_naive))])
        rel = diff / scale
        check(f"[4 det, {label}] parameter gradient finite and == naive",
              all_finite(g_ours) and rel < 1e-9, f"rel diff {rel:.1e}")


def test_position_hessian_matches_naive_formula():
    """Second derivatives with respect to POSITIONS -- the kinetic energy in
    Stage 4 is built from these -- must be finite and correct, including when
    one determinant is singular with rank N-1 (a configuration on its node).

    Rank <= N-2 is excluded on purpose: see report_known_limitation below.
    """
    r = random_positions()
    flat = r.reshape(-1)
    variants = singular_variants(NETS["4 det"])
    del variants["det 0 is exactly zero"]
    for label, params in variants.items():
        h_ours = jax.hessian(lambda x: log_psi(params, x.reshape(N, 2), G).log_abs)(flat)
        h_naive = jax.hessian(lambda x: naive_log_abs(params, x.reshape(N, 2)))(flat)
        diff = worst_error([float(jnp.max(jnp.abs(h_ours - h_naive)))])
        rel = diff / float(jnp.max(jnp.abs(h_naive)))
        check(f"[4 det, {label}] position Hessian finite and == naive",
              all_finite(h_ours) and rel < 1e-8, f"rel diff {rel:.1e}")



def test_slogdet_sum_equals_det_sum():
    """Stage 8: sum_of_determinants_slogdet gives the same log|Psi|, phase and
    parameter gradient as sum_of_determinants at ordinary configurations."""
    def slogdet_log_abs(p, r):
        return sum_of_determinants_slogdet(orbital_matrices(p, r, G)).log_abs
    for label, params in NETS.items():
        r = random_positions()
        a = log_psi(params, r, G)
        b = sum_of_determinants_slogdet(orbital_matrices(params, r, G))
        value = worst_error([abs(float(a.log_abs - b.log_abs)), abs(wrap(float(a.phase - b.phase)))])
        g_det = jax.grad(lambda p: log_psi(p, r, G).log_abs)(params)
        g_slogdet = jax.grad(slogdet_log_abs)(params, r)
        scale = worst_error([float(jnp.max(jnp.abs(x))) for x in jax.tree.leaves(g_det)])
        diff = worst_error([float(jnp.max(jnp.abs(x - y)))
                            for x, y in zip(jax.tree.leaves(g_slogdet), jax.tree.leaves(g_det))])
        check(f"[{label}] slogdet sum == det sum: value and parameter gradient",
              value < TOL and diff / scale < 1e-9, f"value {value:.1e}, gradient rel {diff / scale:.1e}")


def test_slogdet_sum_is_nan_where_one_determinant_vanishes():
    """Stage 8: the price of sum_of_determinants_slogdet, kept visible.  Where
    determinant 0 alone is exactly zero, its parameter gradient is NaN, while
    sum_of_determinants stays finite (and correct:
    test_parameter_gradients_match_naive_formula)."""
    params = singular_variants(NETS["4 det"])["det 0 is exactly zero"]
    r = random_positions()
    g_slogdet = jax.grad(lambda p: sum_of_determinants_slogdet(orbital_matrices(p, r, G)).log_abs)(params)
    g_det = jax.grad(lambda p: log_psi(p, r, G).log_abs)(params)
    check("[4 det, det 0 is exactly zero] slogdet gradient NaN, det gradient finite",
          not all_finite(g_slogdet) and all_finite(g_det))


def report_known_limitation():
    """NOT a pass/fail check -- a measurement, printed so it is never forgotten.

    jnp.linalg.det's second derivative is NaN at rank <= N-2 for N >= 4.  The
    naive formula fails identically, so this is JAX's determinant, not our
    rescaling.  We record whether it still holds, so that a future JAX version
    fixing it would be noticed.
    """
    B = jnp.asarray(np.random.default_rng(4).normal(size=(4, 4)))
    d2 = float(jax.grad(jax.grad(lambda s: jnp.linalg.det(s * B)))(0.0))   # exact answer: 0
    state = "still present" if np.isnan(d2) else f"no longer present (got {d2})"
    print(f"NOTE  known JAX limitation -- det'' at a rank-0 4x4 matrix is NaN: {state}")


ALL_TESTS = [
    test_harness_rejects_nan,
    test_antisymmetry,
    test_pauli_exclusion,
    test_periodicity_per_electron,
    test_log_domain_matches_direct_sum,
    test_one_determinant_is_a_plain_slater_determinant,
    test_log_psi_really_sums_the_determinants,
    test_exactly_zero_psi_gives_minus_inf_not_nan,
    test_parameter_gradients_match_naive_formula,
    test_position_hessian_matches_naive_formula,
    test_slogdet_sum_equals_det_sum,
    test_slogdet_sum_is_nan_where_one_determinant_vanishes,
]

if __name__ == "__main__":
    print(f"jax x64 enabled: {jax.config.jax_enable_x64}\n")
    for test in ALL_TESTS:
        test()
    print("\nall checks passed\n")
    report_known_limitation()
