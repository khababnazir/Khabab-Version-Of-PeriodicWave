"""
The authors' exact code with two sets of settings: what each gives against the paper's Table I.

The four runs in ../runs/ were trained by the authors' released code alone (external/PeriodicWave
at commit 6525d2f3eec5, their training script, their random seed; no line of ours in the training):

    18_electrons_27_sites_released_settings/   their script unchanged                      eps = 5, 10
    18_electrons_27_sites_table2_settings/     a copy of their script with the seven
                                               numbers of the paper's Table II (changes.diff) eps = 5, 10

Each run folder's analysis/ was written by
learn/stage08_own_code/analysis/18_electrons_27_sites/authors_code_results.py (read from learn/,
unchanged): the training energy over the last 20 % of the iterations with a blocking error, and
S(kappa) and the sublattice shares at every checkpoint.  This script puts the four runs side by
side.  It computes no new physics; it CHECKS, and stops if a check fails:
  - the four config.json files that the authors' code wrote: between the two settings exactly the
    seven Table II numbers differ, between eps = 5 and 10 only the interaction strength (and the
    output folder);
  - every run finished (its last iteration is in train_stats.csv), has no non-finite row, and its
    energy is the mean over the last 20 % of the iterations;
  - the checkpoint tables hold as many checkpoints as the summaries say.

Run, from this folder (CPU only):
    $HOME/.local/bin/uv run --no-project --python ../../.venv/bin/python --with matplotlib python settings_analysis.py
    ... settings_analysis.py --refresh     first run learn's authors_code_results.py again on both
                                           run folders (a few minutes), then the comparison
Output: ../results/settings_comparison.csv and ../figures/settings_comparison.png
"""
import argparse
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                         # noqa: E402

HERE = Path(__file__).resolve().parent
FOLDER = HERE.parent
REPO = HERE.parents[1]
LEARN_SCRIPT = REPO / "learn" / "stage08_own_code" / "analysis" / "18_electrons_27_sites" / "authors_code_results.py"
RUNS = {"released": FOLDER / "runs" / "18_electrons_27_sites_released_settings",
        "table2": FOLDER / "runs" / "18_electrons_27_sites_table2_settings"}
NAMES = {"released": "released settings", "table2": "Table II settings"}
COLOURS = {"released": "#1f77b4", "table2": "#e08214"}                  # blue and orange: safe for colour blindness
EPS = (5.0, 10.0)
OUTPUT = "results/2deg-CoulombMoire/CustomPsiformer/el18_0_N27_V4.44370908_45.0_U{}"
U = {5.0: "10.62347336", 10.0: "5.31173668"}                            # the interaction strength in each folder name

# What config.json calls the settings (flattened with dots), and what they are called here.
SETTINGS = {"walkers": "batch_size", "iterations": "optim.iterations", "learning_rate": "optim.lr.rate",
            "learning_rate_delay": "optim.lr.delay", "layers": "network.CustomPsiformer.num_layers",
            "heads": "network.CustomPsiformer.num_heads",
            "perceptrons_per_layer": "network.CustomPsiformer.num_perceptrons_per_layer",
            "determinants": "network.determinants"}
TABLE_II_KEYS = {SETTINGS[name] for name in ("walkers", "iterations", "learning_rate", "learning_rate_delay",
                                             "layers", "heads", "perceptrons_per_layer")}
TABLE_II_VALUES = dict(walkers=4096, iterations=150000, learning_rate=10.0, learning_rate_delay=100000.0,
                       layers=3, heads=6, perceptrons_per_layer=1)     # the paper's Table II
RELEASED_VALUES = dict(walkers=1024, iterations=200000, learning_rate=0.1, learning_rate_delay=20000.0,
                       layers=4, heads=4, perceptrons_per_layer=2)     # their script as published
EPS_KEYS = {"log.save_path", "system.make_local_energy_kwargs.potential_kwargs.interaction_energy_scale"}


def flat(tree, prefix=""):
    """{"optim": {"lr": {"rate": 0.1}}} -> {"optim.lr.rate": 0.1}"""
    out = {}
    for key, value in tree.items():
        name = f"{prefix}.{key}" if prefix else key
        out.update(flat(value, name) if isinstance(value, dict) else {name: value})
    return out


def differing(a, b):
    return {key for key in set(a) | set(b) if a.get(key) != b.get(key)}


def read_configs():
    """The config.json of each run, flattened, after checking what differs between them."""
    configs = {(s, eps): flat(json.loads((RUNS[s] / OUTPUT.format(U[eps]) / "config.json").read_text()))
               for s in RUNS for eps in EPS}
    for eps in EPS:
        assert differing(configs["released", eps], configs["table2", eps]) == TABLE_II_KEYS, eps
    for s in RUNS:
        assert differing(configs[s, 5.0], configs[s, 10.0]) == EPS_KEYS, s
    for s, values in (("released", RELEASED_VALUES), ("table2", TABLE_II_VALUES)):
        for name, value in values.items():
            assert configs[s, 5.0][SETTINGS[name]] == value, (s, name)
    return configs


def refresh():
    """learn's authors_code_results.py, unchanged, on both run folders: rewrites their analysis/."""
    uv = Path(os.environ["HOME"]) / ".local" / "bin" / "uv"
    subprocess.run([str(uv), "run", "--no-project", "--python", str(REPO / ".venv" / "bin" / "python"),
                    "--with", "matplotlib", "--with", "pyblock", "python", LEARN_SCRIPT.name,
                    *(str(folder) for folder in RUNS.values())],
                   cwd=LEARN_SCRIPT.parent, env=dict(os.environ, JAX_PLATFORMS="cpu", CUDA_VISIBLE_DEVICES=""),
                   check=True)


def comparison(configs):
    """One row per run, and the S(kappa) of every checkpoint."""
    rows, checkpoints = [], {}
    for s, folder in RUNS.items():
        summary = pd.read_csv(folder / "analysis" / "summary.csv")
        for eps in EPS:
            found = summary[summary["eps"] == eps]
            assert len(found) == 1, (s, eps)
            row, config = found.iloc[0], configs[s, eps]
            iterations = config[SETTINGS["iterations"]]
            assert bool(row["finished"]) and row["last_step"] == iterations - 1, (s, eps, "not finished")
            assert row["non_finite"] == 0 and row["walkers"] == config[SETTINGS["walkers"]], (s, eps)
            assert row["rows_in_window"] == iterations // 5, (s, eps, "the last 20 %")
            table = pd.read_csv(folder / "analysis" / f"checkpoint_structure_eps{eps:g}.csv")
            assert len(table) == row["checkpoints"], (s, eps, "checkpoints")
            checkpoints[s, eps] = table
            rows.append(dict(settings=s, eps=eps, **{name: config[key] for name, key in SETTINGS.items()},
                             energy=row["energy"], energy_error=row["error"],
                             paper=row["paper"], paper_error=row["paper_error"],
                             minus_paper=row["minus_paper"], minus_paper_in_paper_errors=row["minus_paper"] / row["paper_error"],
                             S_kappa_pooled=row["pooled_S_kappa"], pooled_walkers=row["pooled_walkers"],
                             S_kappa_lowest=table["S_kappa"].min(), S_kappa_highest=table["S_kappa"].max(),
                             sublattice_0=row["pooled_sublattice_0"], sublattice_1=row["pooled_sublattice_1"],
                             sublattice_2=row["pooled_sublattice_2"], checkpoints=row["checkpoints"]))
    out = pd.DataFrame(rows)
    assert np.allclose(out["energy"] - out["paper"], out["minus_paper"])
    return out, checkpoints


def figure(table, checkpoints, path):
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                         "xtick.labelsize": 6, "ytick.labelsize": 6, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 3.0), constrained_layout=True,
                               gridspec_kw=dict(width_ratios=(1, 1.5)))
    # (a) training energy minus the paper's Table I value; grey: the paper's error bar.
    for x, eps in enumerate(EPS):
        error = table.loc[table["eps"] == eps, "paper_error"].iloc[0]
        a.fill_between([x - 0.38, x + 0.38], -error, error, color="0.88", lw=0,
                       label="paper, Table I \u00b1 its error" if x == 0 else None)
        for s, offset in (("released", -0.14), ("table2", 0.14)):
            row = table[(table["eps"] == eps) & (table["settings"] == s)].iloc[0]
            a.errorbar(x + offset, row["minus_paper"], yerr=row["energy_error"], fmt="o", ms=5,
                       color=COLOURS[s], label=NAMES[s] if x == 0 else None)
            a.annotate(f"{row['minus_paper']:+.3f}", (x + offset, row["minus_paper"]), xytext=(0, 6),
                       textcoords="offset points", ha="center", fontsize=6, color=COLOURS[s])
    a.axhline(0, color="0.5", lw=0.6)
    a.set_xticks(range(len(EPS)), [f"\u03b5 = {eps:g}" for eps in EPS])
    a.set_xlim(-0.6, len(EPS) - 0.4)
    a.set_ylim(-0.015, 0.062)
    a.set_ylabel("training energy \u2212 Table I (meV per electron)")
    a.set_title("Released settings land near Table I;\nTable II settings sit higher", loc="left")
    a.legend(frameon=False, loc="upper right")
    # (b) S(kappa) at every checkpoint.
    for s in RUNS:
        for eps, style in ((5.0, "-"), (10.0, "--")):
            t = checkpoints[s, eps]
            b.plot(t["iteration"] / 1000, t["S_kappa"], style, marker="o", ms=2, lw=0.8, color=COLOURS[s],
                   label=f"{NAMES[s]}, \u03b5 = {eps:g}")
    b.set_ylim(0.5, 0.95)
    b.margins(x=0.03)
    b.set_xlabel("iteration (thousands)")
    b.set_ylabel("S(\u03ba) of the walkers at the checkpoint")
    b.set_title("No honeycomb at any checkpoint (a honeycomb gives S(\u03ba) = 4.5)", loc="left")
    b.legend(frameon=False, ncol=2, loc="upper left")
    for ax, letter in ((a, "a"), (b, "b")):
        ax.text(-0.02, 1.02, letter, transform=ax.transAxes, fontsize=10, fontweight="bold", ha="right", va="bottom")
    fig.savefig(path, dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--refresh", action="store_true", help="run learn's authors_code_results.py on both folders first")
    args = parser.parse_args()
    if args.refresh:
        refresh()
    configs = read_configs()
    print("config.json check passed: settings differ in exactly the seven Table II numbers, eps only in the interaction")
    table, checkpoints = comparison(configs)
    (FOLDER / "results").mkdir(exist_ok=True)
    (FOLDER / "figures").mkdir(exist_ok=True)
    table.to_csv(FOLDER / "results" / "settings_comparison.csv", index=False)
    figure(table, checkpoints, FOLDER / "figures" / "settings_comparison.png")
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(table[["settings", "eps", "walkers", "iterations", "learning_rate", "layers", "heads",
                     "perceptrons_per_layer", "energy", "energy_error", "paper", "paper_error", "minus_paper",
                     "minus_paper_in_paper_errors", "S_kappa_pooled", "S_kappa_lowest", "S_kappa_highest",
                     "checkpoints"]].to_string(index=False))


if __name__ == "__main__":
    main()
