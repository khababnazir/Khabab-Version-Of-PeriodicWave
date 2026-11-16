"""
The four authors'-code runs with their final weights FROZEN: energy, density, pair correlation,
S(kappa), sublattices, site occupations -- next to the paper's Table I and our own code.

Reads <run folder>/evaluation/<the authors' output folder name>/ written by
evaluate_authors_weights.py, with learn's analysis functions, unchanged
(learn/stage08_own_code/analysis/18_electrons_27_sites/table1_results.py: load, structure_table,
tiled_mesh, draw_cell; eps3_results.py: energy_table, site_pattern_table), so the numbers are
measured exactly as for our runs.  Our own frozen-weight evaluations (authors' gradient,
Table II settings, released network form) are read the same way, for comparison.

Run, from this folder (CPU only):
    $HOME/.local/bin/uv run --no-project --python ../../.venv/bin/python --with matplotlib --with pyblock python evaluation_results.py
Output: ../results/frozen_energies.csv, frozen_structure.csv, frozen_site_pattern.csv;
        ../figures/frozen_density_pair_correlation.png
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
FOLDER = HERE.parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "learn" / "stage08_own_code" / "analysis" / "18_electrons_27_sites"))
import eps3_results as e3                             # noqa: E402  learn's, unchanged
from matplotlib.colors import TwoSlopeNorm            # noqa: E402

t1 = e3.t1
plt = t1.plt
OUTPUT = Path("results") / "2deg-CoulombMoire" / "CustomPsiformer"
NAME = {5.0: "el18_0_N27_V4.44370908_45.0_U10.62347336", 10.0: "el18_0_N27_V4.44370908_45.0_U5.31173668"}
RUNS = [("released", 5.0), ("table2", 5.0), ("released", 10.0), ("table2", 10.0)]
FOLDERS = {"released": FOLDER / "runs" / "18_electrons_27_sites_released_settings",
           "table2": FOLDER / "runs" / "18_electrons_27_sites_table2_settings"}
LABELS = {"released": "authors' code, released settings", "table2": "authors' code, Table II settings"}
OURS = REPO / "runs" / "stage08" / "18_electrons_27_sites_real_gradient"


def load_authors(settings, eps):
    folder = FOLDERS[settings]
    t1.TRAINING, t1.EVALUATION, t1.GRADIENT = folder / OUTPUT, folder / "evaluation", "real"
    run = t1.load(NAME[eps])
    assert run is not None, f"{folder / 'evaluation' / NAME[eps]}: no finished evaluation"
    assert run["final"] and run["eps"] == eps and run["n_configurations"] == 20_480_000, (settings, eps)
    return dict(run, folder=f"{settings}_eps{eps:g}", settings=settings)


def load_ours(eps):
    t1.DATA, t1.TRAINING, t1.EVALUATION, t1.GRADIENT = OURS, OURS / "training", OURS / "evaluation", "real"
    run = t1.load(f"attention_18e_eps{eps:g}")
    assert run is not None and run["final"]
    return dict(run, folder=f"our_code_eps{eps:g}", settings="our code")


def density_pair_figure(runs, path):
    """Fig. 6's layout, one column per run: density (top), pair correlation (bottom)."""
    fig, axes = plt.subplots(2, len(runs), figsize=(2.4 * len(runs) + 0.9, 5.6), constrained_layout=True)
    densities = [t1.observables.density_map(r["totals"]["density"], r["n_configurations"], r["supercell"]) for r in runs]
    pairs = [t1.observables.pair_correlation_map(r["totals"]["pairs"], r["n_configurations"], t1.N, r["supercell"])
             for r in runs]
    density_style = dict(cmap="magma", vmin=0, vmax=max(m.max() for m in densities))
    pair_style = dict(cmap="RdBu_r", norm=TwoSlopeNorm(vmin=0.0, vcenter=1.0, vmax=max(m.max() for m in pairs)))
    short = {"released": "released settings", "table2": "Table II settings"}
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
                ax.plot(0, 0, "+", color="k", ms=5, mew=0.8)
            ax.set_title(f"{'abcdefgh'[len(runs) * row + column]}   {short[r['settings']]}, \u03b5 = {r['eps']:g}",
                         fontsize=8, loc="left")
            ax.set_xlabel("\u0394x / a$_M$" if centred else "x / a$_M$", fontsize=7)
            if column == 0:
                ax.set_ylabel(("pair correlation\n" if centred else "density\n") +
                              ("\u0394y / a$_M$" if centred else "y / a$_M$"), fontsize=7)
            ax.tick_params(labelsize=6)
            if column == len(runs) - 1:
                fig.colorbar(image, ax=axes[row, :], shrink=0.8,
                             label="n(r)  (a$_M^{-2}$)" if row == 0 else "g(\u0394r)  (1 = independent electrons)")
    fig.suptitle("The authors' own code, final weights frozen (20.5 million configurations each; dashed: the supercell). "
                 "A honeycomb would leave one site in three empty.", x=0.01, ha="left", fontsize=8)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def main():
    runs = [load_authors(*case) for case in RUNS]
    ours = [load_ours(5.0), load_ours(10.0)]
    (FOLDER / "results").mkdir(exist_ok=True)
    (FOLDER / "figures").mkdir(exist_ok=True)
    energies = e3.energy_table(runs + ours)
    paper = {5.0: (-59.127, 0.009), 10.0: (-32.070, 0.007)}
    energies["paper"] = [paper[eps][0] for eps in energies["eps"]]
    energies["paper_error"] = [paper[eps][1] for eps in energies["eps"]]
    energies["frozen_minus_paper"] = energies["E_frozen"] - energies["paper"]
    energies["frozen_minus_training"] = energies["E_frozen"] - energies["E_training_last20pct"]
    structure = t1.structure_table(runs + ours)
    pattern = e3.site_pattern_table(runs + ours)
    energies.to_csv(FOLDER / "results" / "frozen_energies.csv", index=False)
    structure.to_csv(FOLDER / "results" / "frozen_structure.csv", index=False)
    pattern.to_csv(FOLDER / "results" / "frozen_site_pattern.csv", index=False)
    density_pair_figure(runs, FOLDER / "figures" / "frozen_density_pair_correlation.png")
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(energies[["run", "eps", "E_frozen", "E_frozen_error", "E_training_last20pct", "paper", "frozen_minus_paper",
                        "frozen_minus_training", "kinetic", "moire", "coulomb", "halves_apart_in_errors", "settled",
                        "pmove"]].to_string(index=False))
        print(structure.drop(columns=["electrons", "sites", "network"]).to_string(index=False))
        for name, group in pattern.groupby("run", sort=False):
            print(f"{name}: site pattern summed over its 27 wave vectors {group['site_pattern'].sum():.4f}, "
                  f"at +-kappa {group.loc[group['at_kappa'], 'site_pattern'].sum():.4f}")


if __name__ == "__main__":
    main()
