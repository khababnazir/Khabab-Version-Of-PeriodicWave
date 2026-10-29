"""
Stage 8 (our code) -- read our four training runs, compare with the paper's Fig. 5.

The archived analyse_runs.py (learn/archive/stage08_authors_code/analysis),
for the runs that train_moire.py wrote into runs/stage08/training/.  Two
differences: a run is described by its settings.json (network, eps,
gradient) instead of the authors' overrides.json, and its whole history is
one train_stats.csv (on a restart, train_moire.py drops the rows after the
checkpoint and continues the same file).

Run (needs matplotlib and pandas, which the project environment does not have):
    uv run --no-project --with matplotlib --with pandas --with pyblock python analyse_runs.py
(from learn/stage08_own_code/analysis; writes runs/stage08/figures/1_training_curves_vs_fig5.png)

For every run folder in runs/stage08/training/ it
  1. reads the energy of every iteration and converts it to meV per electron,
     as the paper reports it;
  2. averages the last 20% of the iterations, with a blocking analysis
     (pyblock) for the error bar;
  3. checks for drift: the mean of the first half of that window minus the
     mean of the second half.

Why blocking.  Successive iterations are strongly correlated: the walkers move
only a little per step, and the parameters change slowly.  Treating 40,000
correlated values as independent would make the error bar far too small.
Blocking averages neighbouring values into ever larger blocks until the block
means stop being correlated, and reads the error off there.

What the error bar does NOT cover.  Blocking assumes the series is
stationary.  During training the parameters still change, so the energy may
still be drifting down, and blocking cannot remove that bias.  The drift
column helps judge it: a drift several times its error means the run is
still descending and its average is only provisional; a drift of one or two
errors is suggestive, not conclusive.  The frozen-weight evaluation
(evaluate_weights.py) is the cleaner number.

Comparing with Fig. 5.  The paper's dashed lines are also means of the local
energy during VMC, read off the plot by eye.  So the comparison is
approximate, good to about 0.05 meV at best.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyblock
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STAGE08 = Path(__file__).resolve().parents[3] / "runs" / "stage08"
RUNS = STAGE08 / "training"
ENERGY_UNIT_MEV = 3.375559          # hbar^2 / (m* a_M^2); learn/stage04_mcmc moire_units()
N_ELECTRONS = 6
FINAL_FRACTION = 0.2

# Paper, Fig. 5 (3 x 3 supercell, 6 electrons): meV per electron, read off the
# plot by eye, so good to about +-0.05 meV.  Networks named as in train_moire.py.
FIG5 = {(5.0, "slaternet"): -57.9, (5.0, "attention"): -59.5,
        (10.0, "slaternet"): -31.6, (10.0, "attention"): -32.3}
# Correlation energy E(self-attention) - E(SlaterNet), between these named runs
# only: a folder for another experiment (say, with the full gradient) must
# never be paired by accident.
CORRELATION_PAIRS = {5.0: ("attention_eps5", "slaternet_eps5"),
                     10.0: ("attention_eps10", "slaternet_eps10")}
COLOURS = {"attention": "#1f5aa6", "slaternet": "#2e8b57"}
LABELS = {"attention": "self-attention", "slaternet": "SlaterNet"}


def load_history(folder):
    """The run's train_stats.csv as a table, or None if there is no data yet."""
    path = folder / "train_stats.csv"
    if not path.exists() or path.stat().st_size == 0:
        return None
    history = pd.read_csv(path)
    return history if len(history) else None


MINIMUM_SAMPLES = 32        # fewer than this: report no error bar at all


def blocked_mean(values):
    """The mean of `values`, and its standard error from a blocking analysis.

    The mean is always the plain mean of all the values.  Blocking is used
    only for the error bar: at deeper levels pyblock drops a trailing value
    whenever the count is odd, so the mean IT reports can leave samples out.
    With too few samples, or no plateau in the blocking curve, the error is
    reported as unavailable (nan) rather than guessed.
    """
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return float("nan"), float("nan")
    mean = float(np.mean(values))
    if len(values) < MINIMUM_SAMPLES:
        return mean, float("nan")
    stats = pyblock.blocking.reblock(values)
    optimal = pyblock.blocking.find_optimal_block(len(values), stats)[0]
    if np.isnan(optimal):
        return mean, float("nan")
    return mean, float(stats[int(optimal)].std_err)


def describe(folder):
    """Summary of one run, or None while it has no data (e.g. still compiling)."""
    settings = json.loads((folder / "settings.json").read_text())
    history = load_history(folder)
    if history is None:
        return None
    per_electron = history["energy"].to_numpy() * ENERGY_UNIT_MEV / N_ELECTRONS

    window = per_electron[int(len(per_electron) * (1 - FINAL_FRACTION)):]
    mean, error = blocked_mean(window)
    first_half, second_half = np.array_split(window, 2)
    (mean_1, error_1), (mean_2, error_2) = blocked_mean(first_half), blocked_mean(second_half)

    return dict(name=folder.name, network=settings["network"], eps=float(settings["eps"]),
                gradient=settings["gradient"], steps=len(history), history=history,
                per_electron=per_electron, mean=mean, error=error,
                drift=mean_1 - mean_2, drift_error=float(np.hypot(error_1, error_2)))


def main():
    folders = [f for f in sorted(RUNS.iterdir()) if (f / "settings.json").exists()]
    runs = []
    for folder in folders:
        run = describe(folder)
        if run is None:
            print(f"{folder.name}: no data yet")
        else:
            runs.append(run)

    print(f"\n{'run':20s} {'steps':>7s}  {'E/N, last 20% (meV)':>21s}  {'drift (meV)':>17s}  "
          f"{'Fig. 5':>6s}  {'diff':>6s}")
    for run in runs:
        paper = FIG5[(run["eps"], run["network"])]
        print(f"{run['name']:20s} {run['steps']:7d}  {run['mean']:+11.4f} +- {run['error']:.4f}  "
              f"{run['drift']:+8.4f} +- {run['drift_error']:.4f}  {paper:+6.1f}  {run['mean'] - paper:+6.3f}")
    print("  drift = mean of the window's first half minus its second half.  Positive means the")
    print("  second half was lower; it is evidence of continued descent only when it is several")
    print("  times its error, and the error itself assumes each half is stationary.")

    print("\ncorrelation energy per electron, E(self-attention) - E(SlaterNet):")
    by_name = {run["name"]: run for run in runs}
    for eps, (attention_name, slater_name) in CORRELATION_PAIRS.items():
        if attention_name in by_name and slater_name in by_name:
            a, b = by_name[attention_name], by_name[slater_name]
            ours = a["mean"] - b["mean"]
            paper = FIG5[(eps, "attention")] - FIG5[(eps, "slaternet")]
            print(f"  eps = {eps:4.1f}:  {ours:+.3f} +- {np.hypot(a['error'], b['error']):.3f} meV"
                  f"   ({attention_name} - {slater_name};  Fig. 5: about {paper:+.1f})")
        else:
            print(f"  eps = {eps:4.1f}:  waiting for {attention_name} and {slater_name}")

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), constrained_layout=True)
    for ax, eps in zip(axes, (10.0, 5.0)):
        for run in [r for r in runs if r["eps"] == eps]:
            window = max(len(run["per_electron"]) // 200, 1)
            smooth = pd.Series(run["per_electron"]).rolling(window, min_periods=1).mean()
            style = "-" if run["gradient"] == "real" else ":"
            ax.plot(run["history"]["step"], smooth, style, color=COLOURS[run["network"]], lw=1.0,
                    label=f"{LABELS[run['network']]}" + ("" if run["gradient"] == "real" else " (full gradient)"))
        for network in ("slaternet", "attention"):
            ax.axhline(FIG5[(eps, network)], color=COLOURS[network], ls="--", lw=0.8)
        top = FIG5[(eps, "slaternet")] + 1.0
        ax.set_ylim(FIG5[(eps, "attention")] - 0.4, top)
        ax.set_xlabel("training iteration")
        ax.set_ylabel("E/N (meV)")
        ax.set_title(f"eps = {eps:g}   (dashed: paper, Fig. 5)", loc="left", fontsize=8)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=6, frameon=False, loc="upper right")
    out = STAGE08 / "figures" / "1_training_curves_vs_fig5.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=300)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
