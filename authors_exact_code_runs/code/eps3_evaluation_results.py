"""
eps = 3, the authors' code unchanged: the two runs (self-attention and SlaterNet) with their final
weights FROZEN -- energy, density, pair correlation, S(kappa), sublattices, site occupations -- next
to our own code's self-attention run at eps = 3 (runs/stage08/18_electrons_27_sites_eps3_real_gradient:
Table II settings, the authors' gradient).

Reads the evaluations written by evaluate_authors_weights.py (--inverse-eps 1/3, both networks)
with learn's analysis functions, unchanged (table1_results.load, structure_table, tiled_mesh,
draw_cell; eps3_results.energy_table, site_pattern_table), so everything is measured exactly as for
our runs.

Run, from this folder (CPU only):
    $HOME/.local/bin/uv run --no-project --python ../../.venv/bin/python --with matplotlib --with pyblock python eps3_evaluation_results.py
Output, in ../runs/18_electrons_27_sites_eps3_released_settings/:
    results/frozen_energies.csv, frozen_structure.csv, frozen_site_pattern.csv
    figures/frozen_density_pair_correlation.png
"""
import os
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "learn" / "stage08_own_code" / "analysis" / "18_electrons_27_sites"))
import eps3_results as e3                             # noqa: E402  learn's, unchanged
from matplotlib.colors import TwoSlopeNorm            # noqa: E402

t1 = e3.t1
plt = t1.plt
RUN = HERE.parent / "runs" / "18_electrons_27_sites_eps3_released_settings"
NAME = "el18_0_N27_V4.44370908_45.0_U17.70578894"
OURS = REPO / "runs" / "stage08" / "18_electrons_27_sites_eps3_real_gradient"
TITLES = {"authors_attention": "authors' code, self-attention", "authors_slaternet": "authors' code, SlaterNet",
          "our_code_attention": "our code, self-attention"}


def load_authors(network):
    """table1_results.load reads the evaluation EVALUATION / <name> and the training energies
    TRAINING / <name> / train_stats.csv.  The SlaterNet evaluation folder carries the prefix
    "SlaterNet_", the authors' training folder does not; a temporary folder holding a link of that
    name to the training folder bridges the two (nothing is copied or changed)."""
    evaluation = NAME if network == "CustomPsiformer" else f"{network}_{NAME}"
    training = RUN / "results" / "2deg-CoulombMoire" / network / NAME
    with tempfile.TemporaryDirectory() as scratch:
        os.symlink(training, Path(scratch) / evaluation)
        t1.TRAINING, t1.EVALUATION, t1.GRADIENT = Path(scratch), RUN / "evaluation", "real"
        run = t1.load(evaluation)
    assert run is not None, f"{RUN / 'evaluation' / evaluation}: no finished evaluation"
    assert run["final"] and run["eps"] == 3.0 and run["n_configurations"] == 20_480_000, network
    expected = {"CustomPsiformer": "attention", "SlaterNet": "slaternet"}[network]
    assert run["network"] == expected, (network, run["network"])
    return dict(run, folder=f"authors_{expected}")


def load_ours():
    t1.DATA, t1.TRAINING, t1.EVALUATION, t1.GRADIENT = OURS, OURS / "training", OURS / "evaluation", "real"
    run = t1.load("attention_18e_eps3")
    assert run is not None and run["final"] and run["eps"] == 3.0
    return dict(run, folder="our_code_attention")


def density_pair_figure(runs, path):
    """Density (top) and pair correlation (bottom), one column per run, as for all our runs."""
    fig, axes = plt.subplots(2, len(runs), figsize=(2.6 * len(runs) + 1.0, 5.8), constrained_layout=True)
    densities = [t1.observables.density_map(r["totals"]["density"], r["n_configurations"], r["supercell"]) for r in runs]
    pairs = [t1.observables.pair_correlation_map(r["totals"]["pairs"], r["n_configurations"], t1.N, r["supercell"])
             for r in runs]
    density_style = dict(cmap="magma", vmin=0, vmax=max(m.max() for m in densities))
    pair_style = dict(cmap="RdBu_r", norm=TwoSlopeNorm(vmin=0.0, vcenter=1.0, vmax=max(m.max() for m in pairs)))
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
            ax.set_title(f"{'abcdef'[len(runs) * row + column]}   {TITLES[r['folder']]}", fontsize=8, loc="left")
            ax.set_xlabel("\u0394x / a$_M$" if centred else "x / a$_M$", fontsize=7)
            if column == 0:
                ax.set_ylabel(("pair correlation\n" if centred else "density\n") +
                              ("\u0394y / a$_M$" if centred else "y / a$_M$"), fontsize=7)
            ax.tick_params(labelsize=6)
            if column == len(runs) - 1:
                fig.colorbar(image, ax=axes[row, :], shrink=0.8,
                             label="n(r)  (a$_M^{-2}$)" if row == 0 else "g(\u0394r)  (1 = independent electrons)")
    fig.suptitle("\u03b5 = 3, final weights frozen (20.5 million configurations each; dashed: the supercell). "
                 "A honeycomb would leave one site in three empty.", x=0.01, ha="left", fontsize=8)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def main():
    runs = [load_authors("CustomPsiformer"), load_authors("SlaterNet"), load_ours()]
    (RUN / "results").mkdir(exist_ok=True)
    (RUN / "figures").mkdir(exist_ok=True)
    energies = e3.energy_table(runs)
    energies["frozen_minus_training"] = energies["E_frozen"] - energies["E_training_last20pct"]
    structure, pattern = t1.structure_table(runs), e3.site_pattern_table(runs)
    energies.to_csv(RUN / "results" / "frozen_energies.csv", index=False)
    structure.to_csv(RUN / "results" / "frozen_structure.csv", index=False)
    pattern.to_csv(RUN / "results" / "frozen_site_pattern.csv", index=False)
    density_pair_figure(runs, RUN / "figures" / "frozen_density_pair_correlation.png")
    import pandas as pd
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(energies[["run", "E_frozen", "E_frozen_error", "E_training_last20pct", "frozen_minus_training",
                        "kinetic", "moire", "coulomb", "halves_apart_in_errors", "settled", "pmove"]].to_string(index=False))
        print(structure.drop(columns=["electrons", "sites", "eps"]).to_string(index=False))
    for name, group in pattern.groupby("run", sort=False):
        print(f"{name}: site pattern summed over its 27 wave vectors {group['site_pattern'].sum():.4f}, "
              f"at +-kappa {group.loc[group['at_kappa'], 'site_pattern'].sum():.4f}")


if __name__ == "__main__":
    main()
