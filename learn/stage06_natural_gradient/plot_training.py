"""Plot the Stage 6 learning curves: full gradient vs real-part-only, two seeds each.

Run in a throwaway environment so the pinned project .venv is untouched:
    uv run --no-project --with matplotlib --with pandas python plot_training.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

RUNS = Path(__file__).resolve().parents[2] / "runs" / "stage06"
E_UNIT_MEV = 3.375559
COLORS = {"full": "#1f5aa6", "real_only": "#d9822b"}      # blue / orange: colour-blind safe
NAMES = {"full": "full gradient", "real_only": "real part of E_loc only"}

mpl.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
                     "xtick.labelsize": 6, "ytick.labelsize": 6,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titlelocation": "left"})

exact = json.loads((RUNS / "full" / "summary.json").read_text())["exact_energy"]
fig, (ax_e, ax_v) = plt.subplots(1, 2, figsize=(7.2, 3.0), constrained_layout=True)

finals = {}
for estimator in ("full", "real_only"):
    for seed, run in enumerate((estimator, f"{estimator}_seed1")):
        df = pd.read_csv(RUNS / run / "history.csv")
        smooth = df.rolling(100, center=True, min_periods=20).mean()
        style = dict(color=COLORS[estimator], lw=1.3 if estimator == "full" else 1.0,
                     ls="-" if seed == 0 else "--")
        ax_e.plot(df["iteration"], (smooth["energy"] - exact) * E_UNIT_MEV, **style)
        ax_v.plot(df["iteration"], smooth["variance"], **style)
        finals[run] = json.loads((RUNS / run / "summary.json").read_text())

ax_e.axhline(0.0, color="0.4", lw=0.8)
ax_e.text(60, 0.06, "exact", color="0.4", fontsize=6, ha="left", va="bottom")
ax_e.set_yscale("symlog", linthresh=1.0)
ax_e.set_ylim(-0.3, 250)
ax_e.set_xlabel("training iteration")
ax_e.set_ylabel("energy above exact (meV, 6 electrons)")
ax_e.set_title("Real-part-only gradient ends 1-3 meV higher (these runs)")

ax_v.set_yscale("log")
ax_v.set_xlabel("training iteration")
ax_v.set_ylabel("variance of E_loc (code units$^2$)")
ax_v.set_title("Full gradient reaches lower variance (these runs)")

for ax in (ax_e, ax_v):
    ax.set_xlim(0, 3050)

# direct labels, placed in empty regions of each panel (checked by eye on the render)
ax_e.text(1700, 7.0, NAMES["real_only"], color=COLORS["real_only"], fontsize=6, ha="left", va="center")
ax_e.text(250, 0.4, NAMES["full"], color=COLORS["full"], fontsize=6, ha="left", va="center")
ax_v.set_ylim(0.5, None)                      # a clear band below the curves for labels
ax_v.text(1600, 55, NAMES["real_only"], color=COLORS["real_only"], fontsize=6, ha="left", va="center")
ax_v.text(1250, 0.8, NAMES["full"], color=COLORS["full"], fontsize=6, ha="left", va="center")
ax_v.text(0.98, 0.97, "solid: seed 0   dashed: seed 1\n100-iteration running mean",
          transform=ax_v.transAxes, fontsize=6, color="0.3", ha="right", va="top")

out = RUNS / "stage06_training.png"
fig.savefig(out, dpi=300)

renderer = fig.canvas.get_renderer()
texts = [(t, t.get_window_extent(renderer)) for t in fig.findobj(mpl.text.Text)
         if t.get_text().strip() and t.get_visible()]
overlaps = [(a.get_text(), b.get_text()) for i, (a, ba) in enumerate(texts)
            for b, bb in texts[i + 1:] if ba.overlaps(bb)]
print("saved", out, "| text overlaps:", overlaps)
for run, s in finals.items():
    print(f"{run:16s} final {s['final_energy']:+.6f} +- {s['final_error']:.6f}  "
          f"above exact {s['energy_above_exact_meV']:+.3f} meV  variance {s['final_variance']:.3g}")
