"""Prove the Stage 3 tests FAIL on broken output.  Exits non-zero if any
injection slips through.  Run:  python nan_injection_check.py

Two injections, each applied to every check that must detect it:
  A  log|Psi| and phase both NaN
  B  log|Psi| correct, phase NaN  (the case that once passed silently)
"""
import contextlib
import io
import sys

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import slaternet
import test_slaternet as T

real_log_psi = slaternet.log_psi


def all_nan(params, positions, G):
    return slaternet.LogPsi(log_abs=jnp.nan, phase=jnp.nan)


def nan_phase(params, positions, G):
    return slaternet.LogPsi(log_abs=real_log_psi(params, positions, G).log_abs, phase=jnp.nan)


MUST_CATCH = {
    "A: all NaN": (all_nan, [
        T.test_antisymmetry, T.test_pauli_exclusion, T.test_periodicity_per_electron,
        T.test_log_domain_matches_direct_sum, T.test_one_determinant_is_a_plain_slater_determinant,
        T.test_log_psi_really_sums_the_determinants, T.test_exactly_zero_psi_gives_minus_inf_not_nan,
    ]),
    "B: NaN phase only": (nan_phase, [
        T.test_antisymmetry, T.test_periodicity_per_electron,
        T.test_log_domain_matches_direct_sum, T.test_one_determinant_is_a_plain_slater_determinant,
        T.test_log_psi_really_sums_the_determinants, T.test_exactly_zero_psi_gives_minus_inf_not_nan,
    ]),
}

escaped = []
for injection, (broken, tests) in MUST_CATCH.items():
    T.log_psi = broken
    for test in tests:
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                test()
        except AssertionError:
            print(f"caught   [{injection}]  {test.__name__}")
        else:
            print(f"ESCAPED  [{injection}]  {test.__name__}")
            escaped.append((injection, test.__name__))
T.log_psi = real_log_psi

if escaped:
    raise SystemExit(f"{len(escaped)} injection(s) passed undetected: {escaped}")
print(f"\nall {sum(len(t) for _, t in MUST_CATCH.values())} injections caught")
