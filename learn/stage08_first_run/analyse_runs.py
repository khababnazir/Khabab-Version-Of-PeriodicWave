"""
Stage 8 -- read the training runs, compare with the paper's Fig. 5.

Run (needs matplotlib, which the pinned project environment does not have):
    uv run --no-project --with matplotlib --with pandas --with pyblock python analyse_runs.py

For every run folder in runs/stage08/ it
  1. joins the run's history files in the order they were written (see
     load_history) and converts the energy to meV per electron, as the paper
     reports it;
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
column shows how large it is: if it is not small compared with the error bar,
the run has not converged and its average is only provisional.  A
frozen-parameter evaluation is the cleaner number, and is a separate step.

Comparing with Fig. 5.  The paper's dashed lines are also means of the local
energy during VMC.  The caption states its error estimate -- the shaded
region is "the standard deviation of the mean of the local energy averaged
over the batch size per optimization step", a per-step batch error, not a
blocking error over iterations -- but not over which iterations the mean is
taken.  Our values are read off the plot by eye.  So the comparison is
approximate, good to about 0.05 meV at best, and the two error bars measure
different things.
"""
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyblock
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS = Path(__file__).resolve().parents[2] / "runs" / "stage08"
ENERGY_UNIT_MEV = 3.375559          # hbar^2 / (m* a_M^2); learn/stage04_mcmc moire_units()
N_ELECTRONS = 6
FINAL_FRACTION = 0.2

# Paper, Fig. 5 (3 x 3 supercell, 6 electrons): meV per electron, read off the
# plot by eye, so good to about +-0.05 meV.
FIG5 = {(5.0, "SlaterNet"): -57.9, (5.0, "CustomPsiformer"): -59.5,
        (10.0, "SlaterNet"): -31.6, (10.0, "CustomPsiformer"): -32.3}
# Correlation energy E(self-attention) - E(SlaterNet), between these named runs
# only: a folder for another experiment (say, with the complex E_loc) must
# never be paired by accident.
CORRELATION_PAIRS = {5.0: ("psiformer_eps5", "slaternet_eps5"),
                     10.0: ("psiformer_eps10", "slaternet_eps10")}
COLOURS = {"CustomPsiformer": "#1f5aa6", "SlaterNet": "#2e8b57"}
LABELS = {"CustomPsiformer": "self-attention", "SlaterNet": "SlaterNet"}


def load_history(folder):
    """All history files of one run, joined by step.  None if there is no data yet.

    Order of writing.  When a run resumes, the authors' code renames the
    current train_stats.csv to train_stats_N.csv with N one larger than any
    existing number (utils/writers.py, rename_file).  So the order is
    train_stats_1, _2, ..., and train_stats.csv is always the newest.
    Overlap.  After a crash, the old file can hold steps beyond the last
    checkpoint; the resumed run computes those steps again, from the
    checkpoint.  The newer file wins, because its steps belong to the run
    that continues.
    """
    numbered = sorted(glob.glob(str(folder / "train_stats_*.csv")),
                      key=lambda f: int(Path(f).stem.rsplit("_", 1)[1]))
    files = numbered + [str(folder / "train_stats.csv")]
    frames = [pd.read_csv(f) for f in files if Path(f).exists() and Path(f).stat().st_size > 0]
    frames = [frame for frame in frames if len(frame) > 0]
    if not frames:
        return None
    history = pd.concat(frames).drop_duplicates("step", keep="last").sort_values("step")
    return history.reset_index(drop=True)


def blocked_mean(values):
    """Mean and its standard error from a blocking analysis (Flyvbjerg-Petersen)."""
    stats = pyblock.blocking.reblock(np.asarray(values, dtype=float))
    optimal = pyblock.blocking.find_optimal_block(len(values), stats)[0]
    if np.isnan(optimal):          # no plateau found: the series is too short to trust
        return float(np.mean(values)), float("nan")
    block = stats[int(optimal)]
    return float(block.mean), float(block.std_err)


def describe(folder):
    """Summary of one run, or None while it has no data (e.g. still compiling)."""
    config = json.loads((folder / "overrides.json").read_text())
    history = load_history(folder)
    if history is None:
        return None
    network = config["authors_command_line"][-1]
    eps = 1.0 / float(config["authors_command_line"][3])
    per_electron = history["energy"].to_numpy() * ENERGY_UNIT_MEV / N_ELECTRONS

    window = per_electron[int(len(per_electron) * (1 - FINAL_FRACTION)):]
    mean, error = blocked_mean(window)
    first_half, second_half = np.array_split(window, 2)
    (mean_1, error_1), (mean_2, error_2) = blocked_mean(first_half), blocked_mean(second_half)

    return dict(name=folder.name, network=network, eps=round(eps, 6), steps=len(history),
                history=history, per_electron=per_electron, mean=mean, error=error,
                drift=mean_1 - mean_2, drift_error=float(np.hypot(error_1, error_2)),
                laplacian=config["overrides"].get(
                    "system.make_local_energy_kwargs.kinetic_kwargs.laplacian_method", "folx"))


def main():
    folders = [f for f in sorted(RUNS.iterdir()) if (f / "overrides.json").exists()]
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
    print("  drift = mean of the window's first half minus its second half; "
          "positive means still going down.")

    print("\ncorrelation energy per electron, E(self-attention) - E(SlaterNet):")
    by_name = {run["name"]: run for run in runs}
    for eps, (attention_name, slater_name) in CORRELATION_PAIRS.items():
        if attention_name in by_name and slater_name in by_name:
            a, b = by_name[attention_name], by_name[slater_name]
            ours = a["mean"] - b["mean"]
            paper = FIG5[(eps, "CustomPsiformer")] - FIG5[(eps, "SlaterNet")]
            print(f"  eps = {eps:4.1f}:  {ours:+.3f} +- {np.hypot(a['error'], b['error']):.3f} meV"
                  f"   ({attention_name} - {slater_name};  Fig. 5: about {paper:+.1f})")
        else:
            print(f"  eps = {eps:4.1f}:  waiting for {attention_name} and {slater_name}")

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), constrained_layout=True)
    for ax, eps in zip(axes, (10.0, 5.0)):
        for run in [r for r in runs if r["eps"] == eps]:
            window = max(len(run["per_electron"]) // 200, 1)
            smooth = pd.Series(run["per_electron"]).rolling(window, min_periods=1).mean()
            style = "-" if run["laplacian"] == "folx" else ":"
            ax.plot(run["history"]["step"], smooth, style, color=COLOURS[run["network"]], lw=1.0,
                    label=f"{LABELS[run['network']]}" + ("" if run["laplacian"] == "folx" else " (complex E_loc)"))
        for network in ("SlaterNet", "CustomPsiformer"):
            ax.axhline(FIG5[(eps, network)], color=COLOURS[network], ls="--", lw=0.8)
        top = FIG5[(eps, "SlaterNet")] + 1.0
        ax.set_ylim(FIG5[(eps, "CustomPsiformer")] - 0.4, top)
        ax.set_xlabel("training iteration")
        ax.set_ylabel("E/N (meV)")
        ax.set_title(f"eps = {eps:g}   (dashed: paper, Fig. 5)", loc="left", fontsize=8)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=6, frameon=False, loc="upper right")
    out = RUNS / "stage08_fig5.png"
    fig.savefig(out, dpi=300)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
