"""
Stage 8 -- read the training runs, compare with the paper's Fig. 5.

Run (needs matplotlib, which the pinned project environment does not have):
    uv run --no-project --with matplotlib --with pandas --with pyblock python analyse_runs.py

For every run folder in runs/stage08/ it
  1. joins all train_stats*.csv files (a resumed run starts a new one) by step;
  2. converts the energy to meV per electron, as the paper reports it;
  3. estimates the final energy from the last 20% of the iterations, with a
     blocking analysis (pyblock) for the error bar.

Why blocking.  Successive iterations are strongly correlated: the walkers move
only a little per step, and the parameters change slowly.  Treating 40,000
correlated values as independent would make the error bar far too small.
Blocking averages neighbouring values into ever larger blocks until the block
means stop being correlated, and reads the error off there.

What the training average is NOT.  The parameters still change during those
last iterations, so this is the average energy of a slowly changing wave
function.  The paper's Fig. 5 dashed lines are also training averages ("mean
of local energy obtained from the variational Monte Carlo algorithm"), so the
comparison is like for like.  A frozen-parameter evaluation is the cleaner
number, and is a separate step.
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
COLOURS = {"CustomPsiformer": "#1f5aa6", "SlaterNet": "#2e8b57"}
LABELS = {"CustomPsiformer": "self-attention", "SlaterNet": "SlaterNet"}


def load_history(folder):
    """All train_stats*.csv of one run, joined by step (the later file wins on overlap)."""
    files = sorted(glob.glob(str(folder / "train_stats*.csv")), key=lambda f: Path(f).stat().st_mtime)
    frames = [pd.read_csv(f) for f in files if Path(f).stat().st_size > 0]
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
    config = json.loads((folder / "overrides.json").read_text())
    network = config["authors_command_line"][-1]
    eps = 1.0 / float(config["authors_command_line"][3])
    history = load_history(folder)
    per_electron = history["energy"].to_numpy() * ENERGY_UNIT_MEV / N_ELECTRONS
    start = int(len(per_electron) * (1 - FINAL_FRACTION))
    mean, error = blocked_mean(per_electron[start:])
    return dict(name=folder.name, network=network, eps=round(eps, 6), steps=len(history),
                history=history, per_electron=per_electron, mean=mean, error=error,
                laplacian=config["overrides"].get(
                    "system.make_local_energy_kwargs.kinetic_kwargs.laplacian_method", "folx"))


def main():
    runs = [describe(folder) for folder in sorted(RUNS.iterdir())
            if (folder / "overrides.json").exists() and folder.name != "smoke"]
    print(f"{'run':28s} {'steps':>7s}  {'E/N, last 20% (meV)':>22s}  {'Fig. 5':>7s}  {'difference':>10s}")
    for run in runs:
        paper = FIG5[(run["eps"], run["network"])]
        print(f"{run['name']:28s} {run['steps']:7d}  {run['mean']:+12.4f} +- {run['error']:.4f}  "
              f"{paper:+7.1f}  {run['mean'] - paper:+10.3f}")

    print("\ncorrelation energy per electron, E(self-attention) - E(SlaterNet):")
    for eps in (5.0, 10.0):
        pair = {r["network"]: r for r in runs if r["eps"] == eps and r["laplacian"] == "folx"}
        if len(pair) == 2:
            a, b = pair["CustomPsiformer"], pair["SlaterNet"]
            ours = a["mean"] - b["mean"]
            paper = FIG5[(eps, "CustomPsiformer")] - FIG5[(eps, "SlaterNet")]
            print(f"  eps = {eps:4.1f}:  {ours:+.3f} +- {np.hypot(a['error'], b['error']):.3f} meV"
                  f"   (Fig. 5: about {paper:+.1f})")

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
        ax.legend(fontsize=6, frameon=False, loc="upper right")
    out = RUNS / "stage08_fig5.png"
    fig.savefig(out, dpi=300)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
