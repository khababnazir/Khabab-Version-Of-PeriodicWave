"""
PARADOX EXPERIMENT -- the result: the twisted run against the periodic baseline.

Both runs: 18 electrons, 27 sites, eps = 5, self-attention network (released form), Table II,
seed 0, --gradient full.  The only difference is the boundary condition:
    periodic          runs/stage08/18_electrons_27_sites_full_gradient          (twist 0, 0)
    twist_one_third   ../runs/18_electrons_27_sites_twist_one_third_full_gradient  (twist 1/3, 1/3)

Measured on the final weights, frozen, with the SAME functions for both runs.  Nothing is
re-implemented here: the functions are learn's, read from learn/ unchanged --
    table1_results.load, structure_table      energy with blocking error, S(kappa), sublattices
    eps3_results.energy_table, site_pattern_table
    momentum_sectors.measure, weights         weight of the state on each total momentum K
-- except that the wave function is built by THIS folder's train_moire.py, the copy that knows
the twist (momentum_sectors.py imports "train_moire" and "evaluate_weights"; this script imports
this folder's copies first, and checks that these are the ones in use).

Total momentum with a twist.  Moving ALL electrons by a moire lattice vector R leaves the
Hamiltonian and the twisted boundary condition unchanged, so the states still have a total
crystal momentum K, Psi(X + R) = exp(i K . R) Psi(X).  K is the sum of 18 one-electron momenta,
each shifted by k_twist; 18 k_twist = 6 (G1 + G2) = 2 b1 is a moire reciprocal vector, so K is
one of the same 27 momenta as without a twist, and the labels of the two runs can be compared.  Self-check: the determinant of the 18 lowest
twisted plane waves (the closed shell of twist_shells.py) must have weight 1 at K = 0.

Run, from this folder (CPU, a few minutes; it never touches a GPU):
    $HOME/.local/bin/uv run --no-project --python ../../.venv/bin/python --with matplotlib --with pyblock python twist_results.py
Output: printed; ../results/energies.csv, structure.csv, site_pattern.csv, momentum_weights.csv;
../figures/1_density_pair_correlation.png, ../figures/2_training_energy.png.
"""
import os
import sys
from pathlib import Path

os.environ["JAX_PLATFORMS"] = "cpu"                  # before JAX is imported: always off the GPUs
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parent
REPO = HERE.parents[1]
LEARN_ANALYSIS = REPO / "learn" / "stage08_own_code" / "analysis"
sys.path.insert(0, str(LEARN_ANALYSIS / "18_electrons_27_sites"))
sys.path.insert(0, str(LEARN_ANALYSIS))
sys.path.insert(0, str(HERE))
import train_moire                                    # noqa: E402  THIS folder's copy, with the twist
import evaluate_weights                               # noqa: E402  THIS folder's copy
import twist_shells                                   # noqa: E402
import eps3_results as e3                             # noqa: E402  learn's, unchanged
import momentum_sectors as ms                         # noqa: E402  learn's, unchanged
from matplotlib.colors import TwoSlopeNorm            # noqa: E402

t1 = e3.t1                                            # learn's table1_results
plt = t1.plt
assert Path(train_moire.__file__).resolve().parent == HERE, "the trainer must be this folder's copy"
assert Path(evaluate_weights.__file__).resolve().parent == HERE
assert ms.train_moire is train_moire and ms.run_settings is evaluate_weights.run_settings

RUN = "attention_18e_eps5"
PERIODIC = REPO / "runs" / "stage08" / "18_electrons_27_sites_full_gradient"
TWISTED = EXPERIMENT / "runs" / "18_electrons_27_sites_twist_one_third_full_gradient"
SYSTEMS = {"periodic": PERIODIC, "twist_one_third": TWISTED}
THIRD = float(1 / 3)


def load(name):
    """table1_results.load on a system folder; the run is renamed to `name`."""
    data = SYSTEMS[name]
    t1.DATA, t1.TRAINING, t1.EVALUATION, t1.GRADIENT = data, data / "training", data / "evaluation", "full"
    run = t1.load(RUN)
    assert run is not None and run["final"] and run["network"] == "attention", f"{data}: no finished evaluation"
    settings = run["summary"]["training_settings"]
    twist = (settings.get("twist_1", 0.0), settings.get("twist_2", 0.0))
    assert twist == ((THIRD, THIRD) if name == "twist_one_third" else (0.0, 0.0)), (name, twist)
    return dict(run, folder=name, twist=twist)


def same_except_twist(runs):
    """The two evaluations must differ in the twist and in nothing else."""
    # Folders made before 2026-11-08 lack network_form: the released network (train_moire.py fills it in the same way).
    a, b = ({**train_moire.BEFORE_18_ELECTRONS, **r["summary"]["training_settings"]} for r in runs)
    for settings in (a, b):
        settings.pop("twist_1", None)
        settings.pop("twist_2", None)
    assert a == b, {k: (a.get(k), b.get(k)) for k in set(a) | set(b) if a.get(k) != b.get(k)}
    for key in ("network", "eps", "gradient", "seed", "batch_size", "n_electrons", "steps_requested", "burn_in"):
        assert runs[0]["summary"][key] == runs[1]["summary"][key], key


def closed_shell_check(translations, momenta, K):
    """18 twisted plane waves, the closed shell: all weight at total momentum K = 0."""
    k, _, _ = twist_shells.levels(THIRD, THIRD)
    occupied = k[:18]
    X = np.random.default_rng(0).uniform(-3, 3, size=(4, 18, 2))

    def psi(x):
        return np.linalg.det(np.exp(1j * x @ occupied.T))
    ratios = np.array([[psi(x + R) / psi(x) for x in X] for R in translations])
    w, _, imaginary = ms.weights(ratios, K, translations)
    zero = int(np.argmin(np.linalg.norm(K, axis=1)))
    assert np.allclose(K[zero], 0) and np.isclose(w[zero], 1.0, atol=1e-9) and imaginary < 1e-9, w.round(3)
    return zero


def momentum_weights(name, translations, K):
    """Weight on each of the 27 total momenta at the final checkpoint's walkers."""
    ms.STAGE08 = SYSTEMS[name].parent                 # momentum_sectors.measure reads STAGE08/system/training/run
    ratios, done, saved = ms.measure(SYSTEMS[name].name, RUN, translations)
    w, error, imaginary = ms.weights(ratios, K, translations)
    at_zero = np.argmin(np.linalg.norm(translations, axis=1))
    assert np.allclose(ratios[at_zero], 1.0, atol=1e-5), "t(0) = 1"
    assert imaginary < 5 * error.max() + 1e-3, f"weights not real: {imaginary} against errors {error.max()}"
    assert abs(w.sum() - 1) < 1e-4, w.sum()
    return w, error, done, ratios.shape[1]


def density_pair_figure(runs, path):
    """Density (top) and pair correlation (bottom); left periodic, right twisted."""
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.2), constrained_layout=True)
    densities = [t1.observables.density_map(r["totals"]["density"], r["n_configurations"], r["supercell"]) for r in runs]
    pairs = [t1.observables.pair_correlation_map(r["totals"]["pairs"], r["n_configurations"], t1.N, r["supercell"])
             for r in runs]
    density_style = dict(cmap="magma", vmin=0, vmax=max(m.max() for m in densities))
    pair_style = dict(cmap="RdBu_r", norm=TwoSlopeNorm(vmin=0.0, vcenter=1.0, vmax=max(m.max() for m in pairs)))
    titles = {"periodic": "periodic", "twist_one_third": "twist (1/3, 1/3)"}
    for column, r in enumerate(runs):
        bins = r["summary"]["density_bins"]
        for row, (values, centred, style) in enumerate(((densities[column], False, density_style),
                                                         (pairs[column], True, pair_style))):
            ax = axes[row, column]
            for X, Y in t1.tiled_mesh(r["supercell"], bins, centred):
                image = ax.pcolormesh(X, Y, values, shading="flat", rasterized=True, **style)
            xy = t1.draw_cell(ax, r["supercell"], centred)
            ax.set_xlim(xy[:, 0].min() - 0.6, xy[:, 0].max() + 0.6)
            ax.set_ylim(xy[:, 1].min() - 0.6, xy[:, 1].max() + 0.6)
            ax.set_aspect("equal")
            if centred:
                ax.plot(0, 0, "+", color="k", ms=5, mew=0.8)           # the reference electron
            what = "density" if row == 0 else "pair correlation"
            ax.set_title(f"{'abcd'[2 * row + column]}   {what}, {titles[r['folder']]}", fontsize=8, loc="left")
            ax.set_xlabel("\u0394x / a$_M$" if centred else "x / a$_M$")
            ax.set_ylabel("\u0394y / a$_M$" if centred else "y / a$_M$")
            if column == 1:
                fig.colorbar(image, ax=axes[row, :], shrink=0.8,
                             label="n(r)  (a$_M^{-2}$)" if row == 0 else "g(\u0394r)  (1 = independent electrons)")
    fig.suptitle("Self-attention network, 27 sites, 18 electrons, \u03b5 = 5, full gradient, frozen final weights "
                 "(dashed: the supercell)", x=0.02, ha="left", fontsize=8)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def training_figure(runs, path, window=1000):
    """Training energy against iteration: the whole run, and its last third enlarged."""
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), constrained_layout=True)
    colours = {"periodic": "0.35", "twist_one_third": "C3"}
    labels = {"periodic": "periodic", "twist_one_third": "twist (1/3, 1/3)"}
    for r in runs:
        stats = pd.read_csv(SYSTEMS[r["folder"]] / "training" / RUN / "train_stats.csv", usecols=["step", "energy"])
        smooth = (stats["energy"] * t1.ENERGY_UNIT_MEV / t1.N).rolling(window).mean()
        for ax in axes:
            ax.plot(stats["step"] / 1000, smooth, color=colours[r["folder"]], lw=0.9, label=labels[r["folder"]])
    axes[0].set_ylim(-59.2, -58.0)
    axes[1].set_xlim(100, 150)
    axes[1].set_ylim(-59.12, -59.02)
    for ax, letter in zip(axes, "ab"):
        ax.set_xlabel("iteration (thousands)")
        ax.set_ylabel("energy (meV per electron)")
        ax.set_title(f"{letter}   {'whole run' if letter == 'a' else 'last third, enlarged'}", fontsize=9, loc="left")
    axes[0].legend(frameon=False)
    fig.suptitle(f"Training energy, {window}-iteration moving average (periodic and twisted energies are not "
                 "expected to be equal)", x=0.02, ha="left", fontsize=8)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def main():
    runs = [load("periodic"), load("twist_one_third")]
    same_except_twist(runs)
    results, figures = EXPERIMENT / "results", EXPERIMENT / "figures"
    results.mkdir(exist_ok=True)
    figures.mkdir(exist_ok=True)

    energies, structure, pattern = e3.energy_table(runs), t1.structure_table(runs), e3.site_pattern_table(runs)
    energies.to_csv(results / "energies.csv", index=False)
    structure.to_csv(results / "structure.csv", index=False)
    pattern.to_csv(results / "site_pattern.csv", index=False)

    translations, momenta, K, shell = ms.geometry()
    zero = closed_shell_check(translations, momenta, K)
    print("self-check passed: the 18 lowest twisted plane waves have all their weight at K = 0")
    rows, weights = [], {}
    for r in runs:
        w, error, done, n_walkers = momentum_weights(r["folder"], translations, K)
        assert done == r["step"], "the weights and the evaluation must use the same checkpoint"
        weights[r["folder"]] = (w, error)
        for i in range(27):
            rows.append(dict(run=r["folder"], checkpoint=done, walkers=n_walkers, m1=momenta[i, 0], m2=momenta[i, 1],
                             K=shell[i], weight=w[i], error=error[i]))
    pd.DataFrame(rows).to_csv(results / "momentum_weights.csv", index=False)

    density_pair_figure(runs, figures / "1_density_pair_correlation.png")
    training_figure(runs, figures / "2_training_energy.png")

    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(energies[["run", "checkpoint", "E_frozen", "E_frozen_error", "E_training_last20pct",
                        "E_training_last20pct_error", "kinetic", "moire", "coulomb", "halves_apart_in_errors",
                        "settled", "sigma_local_energy_meV", "pmove"]].to_string(index=False))
        print(structure.drop(columns=["electrons", "sites", "network", "eps"]).to_string(index=False))
        for name, group in pattern.groupby("run", sort=False):
            print(f"{name}: site pattern summed over its 27 wave vectors {group['site_pattern'].sum():.4f}; "
                  f"at +-kappa {group.loc[group['at_kappa'], 'site_pattern'].sum():.4f}; largest three:")
            print(group.sort_values("site_pattern", ascending=False).head(3)[["m1", "m2", "q_bz", "site_pattern"]]
                  .to_string(index=False))
    for name, (w, error) in weights.items():
        print(f"{name}: weight per shell of total momentum |K|:  " + "   ".join(
            f"{s:.3f}: {w[shell == s].sum():+.3f}({np.sqrt((error[shell == s] ** 2).sum()):.3f})" for s in sorted(set(shell))))
        order = np.argsort(-w)[:4]
        print("   largest single momenta (m1, m2 in units of b1, b2):  " + "   ".join(
            f"({momenta[i, 0]:+.3f}, {momenta[i, 1]:+.3f}) |K| {shell[i]:.3f}: {w[i]:+.3f}({error[i]:.3f})" for i in order)
              + f"      at K = 0: {w[zero]:+.3f}({error[zero]:.3f}); smallest anywhere {w.min():+.3f}")


if __name__ == "__main__":
    main()
