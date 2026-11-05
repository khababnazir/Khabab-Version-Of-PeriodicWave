"""
Stage 8 (our code) -- Table I with the authors' gradient against the full gradient.

Run after `table1_results.py` and `table1_results.py --gradient full` (CPU is enough), from
learn/stage08_own_code/analysis/18_electrons_27_sites:
    uv run --no-project --python ../../../../.venv/bin/python --with matplotlib --with pyblock python compare_gradients.py

Reads   runs/stage08/18_electrons_27_sites_{real,full}_gradient/results/table1.csv, correlation_energy.csv
        runs/stage08/18_electrons_27_sites_{real,full}_gradient/training/<run>/train_stats.csv
Writes  runs/stage08/18_electrons_27_sites_full_gradient/results/real_vs_full.csv
        runs/stage08/18_electrons_27_sites_full_gradient/results/correlation_real_vs_full.csv
        runs/stage08/18_electrons_27_sites_full_gradient/figures/5_real_vs_full.png

The two sets differ only in the gradient: the same seed, settings and code, so the same start
(iteration 0 is identical in both).  "real" is the authors' convention, the reproduction;
"full" keeps the term 2 < Im d  d phase/d theta >.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                     # noqa: E402

HERE = Path(__file__).resolve().parent
STAGE8 = HERE.parents[3] / "runs" / "stage08"
FOLDER = {g: STAGE8 / f"18_electrons_27_sites_{g}_gradient" for g in ("real", "full")}
RUNS = ["attention_18e_eps10", "attention_18e_eps5", "slaternet_18e_eps10", "slaternet_18e_eps5"]
TITLES = {"attention_18e_eps10": "self-attention, \u03b5 = 10", "attention_18e_eps5": "self-attention, \u03b5 = 5",
          "slaternet_18e_eps10": "SlaterNet, \u03b5 = 10", "slaternet_18e_eps5": "SlaterNet, \u03b5 = 5"}
TO_MEV = 3.375559 / 18                               # code units (whole system) -> meV per electron
WINDOW = 1000
COLOURS = {"real": "0.45", "full": "#c2410c"}
NAMES = {"real": "authors' gradient (real part)", "full": "full gradient"}


def energies():
    tables = {g: pd.read_csv(FOLDER[g] / "results" / "table1.csv").set_index("run") for g in FOLDER}
    rows = []
    for run in RUNS:
        r, f = tables["real"].loc[run], tables["full"].loc[run]
        assert r["checkpoint"] == f["checkpoint"] == 150000, run
        rows.append(dict(
            run=run, electrons=18, sites=27, network=r["network"], eps=r["eps"],
            E_frozen_real=r["E_frozen"], E_frozen_real_error=r["E_frozen_error"],
            E_frozen_full=f["E_frozen"], E_frozen_full_error=f["E_frozen_error"],
            full_minus_real=f["E_frozen"] - r["E_frozen"],
            full_minus_real_error=float(np.hypot(r["E_frozen_error"], f["E_frozen_error"])),
            E_training_last20pct_real=r["E_training_last20pct"], E_training_last20pct_full=f["E_training_last20pct"],
            E_paper_table1=r["E_paper_table1"], E_paper_table1_error=r["E_paper_table1_error"],
            real_minus_paper=r["frozen_minus_paper"], full_minus_paper=f["frozen_minus_paper"],
            E_bp_ed=r["E_bp_ed"]))
    return pd.DataFrame(rows)


def correlation():
    c = {g: pd.read_csv(FOLDER[g] / "results" / "correlation_energy.csv").set_index("eps") for g in FOLDER}
    return pd.DataFrame([dict(electrons=18, sites=27, eps=eps, real=c["real"].loc[eps, "ours"],
                              real_error=c["real"].loc[eps, "ours_error"], full=c["full"].loc[eps, "ours"],
                              full_error=c["full"].loc[eps, "ours_error"], paper=c["real"].loc[eps, "paper"])
                         for eps in sorted(c["real"].index, reverse=True)])


def curve(gradient, run):
    stats = pd.read_csv(FOLDER[gradient] / "training" / run / "train_stats.csv", usecols=["step", "energy"])
    assert stats["step"].is_unique and len(stats) == 150000, (gradient, run)
    window = stats["step"] // WINDOW
    means = stats.groupby(window)["energy"].mean() * TO_MEV
    return (means.index.to_numpy() * WINDOW + WINDOW / 2) / 1000, means.to_numpy()


def figure(table, path):
    fig = plt.figure(figsize=(7.2, 5.4), constrained_layout=True)
    grid = fig.add_gridspec(2, 4, height_ratios=[1.25, 1])
    for k, run in enumerate(RUNS):
        ax = fig.add_subplot(grid[0, k])
        row = table.set_index("run").loc[run]
        paper, paper_error = row["E_paper_table1"], row["E_paper_table1_error"]
        ax.axhspan(paper - paper_error, paper + paper_error, color="0.88", lw=0)
        ax.axhline(paper, color="black", lw=0.7, ls="--")
        tail = []
        for g in ("real", "full"):
            x, y = curve(g, run)
            keep = x >= 10
            ax.plot(x[keep], y[keep], color=COLOURS[g], lw=0.9, label=NAMES[g])
            ax.plot(150, row[f"E_frozen_{g}"], "o", color=COLOURS[g], mec="black", mew=0.5, ms=4, zorder=5,
                    clip_on=False)
            tail.append(y[x >= 20])
        low = min(min(t.min() for t in tail), paper - paper_error)
        high = max(t.max() for t in tail)
        pad = 0.08 * (high - low)
        ax.set_ylim(low - pad, high + pad)
        ax.set_xlim(10, 155)
        network, eps = TITLES[run].split(", ")
        ax.set_title(f"{'abcd'[k]}   {network}\n      {eps}", loc="left", fontsize=7.5)
        ax.set_xlabel("iteration (thousands)")
        if k == 0:
            ax.set_ylabel("E/N (meV)")
        ax.ticklabel_format(axis="y", useOffset=False)
    handles, labels = ax.get_legend_handles_labels()
    handles += [plt.Line2D([], [], color="black", lw=0.7, ls="--"),
                plt.Line2D([], [], marker="o", color="0.6", mec="black", mew=0.5, ls="", ms=4)]
    labels += ["paper's Table I and\nits error bar (grey)", "frozen weights\nat 150,000"]

    ax = fig.add_subplot(grid[1, :3])
    ax.axvline(0, color="black", lw=0.7, ls="--")
    for y, (_, row) in enumerate(table.iterrows()):
        ax.fill_betweenx([y - 0.35, y + 0.35], -row["E_paper_table1_error"], row["E_paper_table1_error"],
                         color="0.88", lw=0)
        for g, dy in (("real", 0.12), ("full", -0.12)):
            ax.errorbar(row[f"{g}_minus_paper"], y + dy, xerr=row[f"E_frozen_{g}_error"], fmt="o", ms=4,
                        color=COLOURS[g], mec="black", mew=0.5)
            ax.annotate(f"{row[f'{g}_minus_paper']:+.3f}".replace("-", "\u2212"),
                        (row[f"{g}_minus_paper"], y + dy), xytext=(6, 0), textcoords="offset points",
                        va="center", fontsize=6.5, color=COLOURS[g])
    shifts = table[["real_minus_paper", "full_minus_paper"]].to_numpy()
    ax.set_xlim(min(shifts.min(), -table["E_paper_table1_error"].max()) - 0.02, shifts.max() + 0.04)
    ax.set_yticks(range(len(table)), [TITLES[r] for r in table["run"]])
    ax.set_ylim(len(table) - 0.5, -0.5)
    ax.set_xlabel("frozen-weight E/N \u2212 paper's Table I (meV)   (right = higher energy)")
    ax.set_title("e   frozen weights at 150,000 against the paper", loc="left", fontsize=8)
    legend_ax = fig.add_subplot(grid[1, 3])
    legend_ax.axis("off")
    legend_ax.legend(handles, labels, loc="center left", frameon=False, fontsize=7)
    fig.suptitle("Table I, 18 electrons on 27 sites: the authors' gradient against the full gradient "
                 "(same seed, settings and code)", x=0.01, ha="left", fontsize=8.5)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def main():
    results, figures = FOLDER["full"] / "results", FOLDER["full"] / "figures"
    table, corr = energies(), correlation()
    table.to_csv(results / "real_vs_full.csv", index=False)
    corr.to_csv(results / "correlation_real_vs_full.csv", index=False)
    with pd.option_context("display.width", 220, "display.precision", 4):
        print(table.T.to_string())
        print(corr.to_string(index=False))
    with plt.rc_context({"font.size": 7.5, "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7}):
        figure(table, figures / "5_real_vs_full.png")
    print(f"written: {results / 'real_vs_full.csv'}, {results / 'correlation_real_vs_full.csv'}, "
          f"{figures / '5_real_vs_full.png'}")


if __name__ == "__main__":
    main()
