"""
Stage 8 -- which total momenta does a trained wave function hold?  (18 electrons, 27 sites)

Moving every electron by the same moire lattice vector R changes neither the moire potential
nor the kinetic or Coulomb energy.  So the eigenstates of the Hamiltonian, WITH the interaction,
can be labelled by a total crystal momentum K, one of the 27 momenta of the cell:
    Psi_K(X + R) = exp(i K . R) Psi_K(X).
A trained network need not be one of them.  It is a sum  Psi = sum_K c_K Psi_K  with weights
w_K = |c_K|^2 that add up to 1.  With samples X of |Psi|^2,
    t(R) = < Psi(X + R) / Psi(X) > = sum_K w_K exp(i K . R),     w_K = (1/27) sum_R exp(-i K . R) t(R),
the sum over the 27 lattice vectors R of the cell.

Why (2026-11-10).  shell_structure.py shows that 18 electrons WITHOUT interaction leave one
place free in the shell of six momenta at |k| = 2.793, and that our uneven site patterns lie at
the wave vectors between that shell's members.  That is a statement about one electron.  This
script asks the trained, interacting wave function directly.  If it is a liquid that comes from
that open shell, its total momentum is minus the momentum left out, so its weight sits on the
six momenta of the 2.793 shell; a coherent sum of several of them has an uneven density (a
single one has an even density).  Weight elsewhere means the shell picture does not describe it.
A honeycomb in the density needs weight on momenta that differ by kappa (an ideal one that stays
put: three such momenta, 1/3 each); no two momenta of the 2.793 shell differ by kappa, so weight
in that shell alone cannot show a honeycomb in the density.

What the weights do not say: whether these momenta hold the ground state, or whether the part
of Psi in each momentum is the best possible one.  The Hamiltonian does not mix different K, so
the energy of Psi is the weighted average of the energies of its parts.

Self-checks at every run: a determinant of 18 plane waves (the lowest 13 momenta and five of
the 2.793 shell) must give weight 1 at minus the momentum left out, for every configuration;
t(0) = 1; the weights are real within their errors.

Run, from this folder (CPU, a few minutes; it never touches a GPU):
    uv run --no-project --python ../../../../.venv/bin/python --with matplotlib --with pyblock python momentum_sectors.py
Output: printed, and results/momentum_sectors.csv in the eps = 3 folder (beside shell_structure.csv).
"""
import os
import sys
from pathlib import Path

os.environ["JAX_PLATFORMS"] = "cpu"                  # before JAX is imported: always off the GPUs
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))                  # evaluate_weights.py, observables.py
from evaluate_weights import run_settings, checkpoint_path   # noqa: E402  (puts train_moire.py on the path)
import train_moire                                    # noqa: E402
import observables                                    # noqa: E402
import slaternet                                      # noqa: E402
from shell_structure import B, allowed_momenta, reduce_to_zone, lowest_band   # noqa: E402  (this folder)

N = 18
STAGE08 = HERE.parents[3] / "runs" / "stage08"
EPS3 = STAGE08 / "18_electrons_27_sites_eps3_real_gradient"
RUNS = [("18_electrons_27_sites_real_gradient", "attention_18e_eps10"),
        ("18_electrons_27_sites_real_gradient", "attention_18e_eps5"),
        ("18_electrons_27_sites_eps3_real_gradient", "attention_18e_eps3"),
        ("18_electrons_27_sites_full_gradient", "attention_18e_eps5"),
        ("18_electrons_27_sites_paper_network_real_gradient", "attention_18e_eps5"),
        ("18_electrons_27_sites_real_gradient", "slaternet_18e_eps5")]


def geometry():
    supercell = np.asarray(slaternet.supercell_vectors(27), dtype=float)
    translations = np.asarray(observables.moire_sites(supercell))            # the 27 lattice vectors R
    momenta = allowed_momenta(27)                                           # (27, 2) in units of b1, b2
    K = momenta @ B.T                                                       # Cartesian
    shell = np.array([round(reduce_to_zone(m)[1], 3) for m in momenta])
    assert len(translations) == 27 and np.any(np.all(np.isclose(translations, 0), axis=1)), "R = 0 must be one of them"
    # exp(i K . R) must be an orthogonal table, or the momenta and translations do not belong together.
    table = np.exp(1j * K @ translations.T)
    assert np.allclose(table @ table.conj().T, 27 * np.eye(27), atol=1e-8)
    return translations, momenta, K, shell


def weights(ratios, K, translations):
    """ratios[R, walker] = Psi(X + R) / Psi(X).  Returns w_K, its error, the largest |imaginary part|."""
    per_walker = np.exp(-1j * K @ translations.T) @ ratios / 27              # (27 momenta, walkers)
    w = per_walker.mean(axis=1)
    error = per_walker.real.std(axis=1, ddof=1) / np.sqrt(ratios.shape[1])
    return w.real, error, float(np.abs(w.imag).max())


def plane_wave_check(translations, momenta, K, shell):
    """18 plane waves, the 2.793 shell with one momentum left out: all weight at minus that momentum."""
    rng = np.random.default_rng(0)
    order = np.argsort([lowest_band(k) for k in K], kind="stable")
    assert np.allclose(shell[order[13:19]], 2.793, atol=1e-3)
    left_out = order[18]
    occupied = K[order[:18]]
    X = rng.uniform(-3, 3, size=(4, N, 2))

    def psi(x):
        return np.linalg.det(np.exp(1j * x @ occupied.T))                     # det over (electron, orbital)
    ratios = np.array([[psi(x + R) / psi(x) for x in X] for R in translations])
    w, _, imaginary = weights(ratios, K, translations)
    expected = np.argmin(np.linalg.norm([reduce_to_zone(np.linalg.solve(B, k + K[left_out]))[0] for k in K], axis=1))
    assert np.isclose(w[expected], 1.0, atol=1e-9) and imaginary < 1e-9, (w.round(3), expected)
    assert np.isclose(shell[expected], 2.793, atol=1e-3)
    return left_out, expected


def measure(system, name, translations):
    """ratios[R, walker] for the newest checkpoint of a run, and that checkpoint's iteration."""
    import jax
    jax.config.update("jax_default_matmul_precision", "float32")              # as the training
    run = STAGE08 / system / "training" / name
    saved, training_settings = run_settings(run)
    final = checkpoint_path(run, saved)
    if not final.exists():                                                    # a run still training
        final = train_moire.checkpoints_newest_first(run)[0]
    problem = train_moire.make_problem(saved["network"], saved["eps"], saved["electrons"], training_settings)
    like = problem.init(jax.random.PRNGKey(0), problem.arch)
    done, params, walkers, _ = train_moire.load_parameters(final, like)

    @jax.jit
    def log_psi(x):
        out = jax.vmap(lambda R: problem.log_psi(params, R))(x)
        return out.log_abs, out.phase
    log_abs0, phase0 = (np.asarray(a, dtype=np.float64) for a in log_psi(walkers))
    ratios = []
    for R in translations:
        log_abs, phase = (np.asarray(a, dtype=np.float64) for a in log_psi(walkers + jax.numpy.asarray(R, walkers.dtype)))
        ratios.append(np.exp(log_abs - log_abs0 + 1j * (phase - phase0)))
    return np.array(ratios), done, saved


def main():
    translations, momenta, K, shell = geometry()
    left_out, expected = plane_wave_check(translations, momenta, K, shell)
    print(f"self-check passed: 18 plane waves without the momentum {momenta[left_out].round(3)} have all their weight "
          f"at K = {momenta[expected].round(3)} (minus it), in the 2.793 shell")
    shells = sorted(set(shell))
    rows = []
    for system, name in RUNS:
        ratios, done, saved = measure(system, name, translations)
        w, error, imaginary = weights(ratios, K, translations)
        at_zero = np.argmin(np.linalg.norm(translations, axis=1))
        assert np.allclose(ratios[at_zero], 1.0, atol=1e-5), "t(0) = 1"
        assert imaginary < 5 * error.max() + 1e-3, f"weights not real: {imaginary} against errors {error.max()}"
        form = saved["settings"].get("network_form", "released")
        label = f"{name} ({form} network, {saved['gradient']} gradient, iteration {done})"
        by_shell = {s: (w[shell == s].sum(), np.sqrt((error[shell == s] ** 2).sum())) for s in shells}
        print(label)
        print("   weight per shell of total momentum:  " +
              "   ".join(f"|K| {s:.3f}: {by_shell[s][0]:+.3f}({by_shell[s][1]:.3f})" for s in shells))
        members = np.flatnonzero(np.isclose(shell, 2.793, atol=1e-3))
        print("   the six momenta of the 2.793 shell:  " +
              "  ".join(f"{w[i]:+.3f}({error[i]:.3f})" for i in members) +
              f"      smallest weight anywhere {w.min():+.3f}; largest |imaginary part| {imaginary:.3f}")
        for i in range(27):
            rows.append(dict(system=system, run=name, network_form=form, gradient=saved["gradient"], eps=saved["eps"],
                             checkpoint=done, m1=momenta[i, 0], m2=momenta[i, 1], K=shell[i], weight=w[i], error=error[i]))
    pd.DataFrame(rows).to_csv(EPS3 / "results" / "momentum_sectors.csv", index=False)


if __name__ == "__main__":
    main()
