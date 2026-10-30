"""Draw the five figures of speed_study_report.pdf from speed_results.jsonl (Stage 8 speed study).

Run from this folder:  python make_figures.py      (needs matplotlib and numpy; writes fig*.png here)
Figures C, D and E use measured numbers; the numbers that are not in speed_results.jsonl
(earlier 18-electron pieces, the authors' timings) are written below with where they come from.
"""
import json
import os

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt                     # noqa: E402
import numpy as np                                  # noqa: E402
from matplotlib.patches import Patch, Rectangle     # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
mpl.rcParams.update({
    "font.family": "sans-serif", "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
    "legend.fontsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.6,
    "xtick.direction": "out", "ytick.direction": "out", "xtick.major.size": 3, "ytick.major.size": 3,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "figure.dpi": 200, "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.titlelocation": "left", "lines.linewidth": 1.2, "patch.linewidth": 0.6})
BLUE, ORANGE, GREY, GREEN, RED = "#1f5fa8", "#d9822b", "#b8b8b8", "#2a9d8f", "#c0392b"
R = {json.loads(l)["run"]: json.loads(l)["result"]
     for l in open(os.path.join(HERE, "speed_results.jsonl")).read().strip().splitlines()}


def save(fig, name):
    fig.savefig(os.path.join(HERE, name), dpi=200)
    plt.close(fig)


# ---- Figure A (report Figure 1): the Laplacian needs only the diagonal of the Hessian
fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.6))
for ax, n_el in zip(axes, [6, 18]):
    d = 2 * n_el
    for i in range(d):
        for j in range(d):
            ax.add_patch(Rectangle((j, d - 1 - i), 1, 1, facecolor=ORANGE if i == j else "#f1f1f1",
                                   edgecolor="white", linewidth=0.4 if d > 20 else 0.8))
    ax.set_xlim(-0.5, d + 0.5); ax.set_ylim(-0.5, d + 0.5); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(f"{n_el} electrons: {d} coordinates\n{d * d:,} second derivatives computed, {d} needed", loc="left")
axes[0].text(0, -2.2, "orange = the diagonal (what the Laplacian adds up)\n"
             "grey = computed by the full-Hessian route, then thrown away", fontsize=8, va="top")
fig.tight_layout()
save(fig, "figA_hessian_vs_laplacian.png")

# ---- Figure B (report Figure 5): derivative of det vs derivative of log|det| near a singular matrix
e = np.logspace(-7, 0, 200)
fig, ax = plt.subplots(figsize=(5.4, 3.5))
ax.loglog(e, np.ones_like(e), color=BLUE, lw=2)
ax.loglog(e, 1 / e, color=ORANGE, lw=2)
ax.invert_xaxis()
ax.set_ylim(0.3, 3e7)
ax.text(0.03, 0.80, "d(log|det A|)/dε = 1/ε\nslogdet's derivative: grows without limit", color=ORANGE, fontsize=8,
        transform=ax.transAxes, va="top")
ax.text(0.97, 0.13, "d(det A)/dε = 1\nplain det's derivative: stays finite", color=BLUE, fontsize=8,
        transform=ax.transAxes, ha="right", va="bottom")
ax.set_xlabel("ε, how far A = [[1, 1], [1, 1+ε]] is from singular (det A = ε)", labelpad=6)
ax.set_ylabel("size of the derivative")
ax.set_title("As the matrix becomes singular, only the logarithm's derivative blows up", loc="left")
fig.tight_layout()
save(fig, "figB_singular_derivative.png")

# ---- Figure C (report Figure 2): the determinant step alone, 1024 walkers
routes = [("folx_det", "folx + plain det (Stage 3)"), ("folx_slogdet", "folx + slogdet (authors' way)"),
          ("hessian_det", "full Hessian + plain det"), ("hessian_slogdet", "full Hessian + slogdet")]
fig, ax = plt.subplots(figsize=(6.6, 3.0))
for y, (key, label) in enumerate(routes[::-1]):
    for n, col, dy in [(6, GREY, 0.14), (18, BLUE, -0.14)]:
        r = R[f"det_alone_{n}e_{key}_b1024"]
        if "error" in r:   # failed run: show the estimate (64-walker value x 16), marked as a failure
            est = R[f"det_alone_{n}e_{key}_b64"]["compiled_scratch_GiB"] * 16
            ax.scatter([est], [y + dy], marker="x", s=60, color=RED, zorder=3)
            ax.text(16.5, y - 0.03, f"failed: \"No valid config found\"\n(≈{est:.0f} GiB expected)", color=RED,
                    fontsize=7, va="top", ha="right")
        else:
            v = r["compiled_scratch_GiB"]
            ax.scatter([v], [y + dy], s=36, color=col, zorder=3)
            ax.text(v * 1.25, y + dy, f"{v:.2g} GiB" if v >= 0.1 else f"{v * 1024:.0f} MiB", fontsize=7,
                    va="center", color=col)
ax.axvline(24, color="black", lw=0.8, ls="--")
ax.text(24 * 1.08, -0.42, "24 GiB = one RTX 3090", fontsize=7, va="bottom")
ax.set_xscale("log"); ax.set_xlim(0.01, 400)
ax.set_xticks([0.01, 0.1, 1, 10, 100]); ax.set_xticklabels(["10 MiB", "100 MiB", "1 GiB", "10 GiB", "100 GiB"])
ax.set_yticks(range(4)); ax.set_yticklabels([l for _, l in routes[::-1]]); ax.set_ylim(-0.5, 3.6)
ax.set_xlabel("scratch memory for the determinant step alone, 1024 walkers")
ax.scatter([], [], color=GREY, s=36, label="6 electrons"); ax.scatter([], [], color=BLUE, s=36, label="18 electrons")
ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.0), fontsize=7)
ax.set_title("folx + plain det is the only route that explodes at 18 electrons", loc="left")
fig.tight_layout()
save(fig, "figC_determinant_alone.png")

# ---- Figure D (report Figure 3): local energy of the real attention network, 18 e, 1024 walkers.
# Measured earlier with pieces18.py (logs in earlier/): ours and the authors', one GPU.
rows = [("ours: full Hessian + plain det (before)", 0.476, 16.9, BLUE),
        ("ours: full Hessian + slogdet", 0.4316, 16.13, BLUE),
        ("ours: folx + plain det", None, None, BLUE),
        ("ours: folx + slogdet (after)", 0.1296, 1.37, GREEN),
        ("authors: folx + slogdet", 0.1305, 1.37, ORANGE)]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.9), sharey=True)
ys = np.arange(len(rows))[::-1]
for y, (lab, t, m, col) in zip(ys, rows):
    for ax, v, unit in [(a1, t, "s"), (a2, m, "GiB")]:
        if v is None:
            ax.text(0.005 if unit == "s" else 0.18, y, "failed: \"No valid config found\"", color=RED, fontsize=7,
                    va="center")
        else:
            ax.barh(y, v, color=col, height=0.6)
            ax.text(v + (0.008 if unit == "s" else 0.3), y, f"{v:g} {unit}", va="center", fontsize=7)
a1.set_yticks(ys); a1.set_yticklabels([r[0] for r in rows])
a1.set_xlim(0, 0.6); a2.set_xlim(0, 21)
a1.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5]); a2.set_xticks([0, 5, 10, 15, 20])
a1.set_xlabel("time per call (s)"); a2.set_xlabel("peak GPU memory (GiB)")
a1.set_title("Local energy of 1024 walkers, attention network, 18 electrons", loc="left")
fig.tight_layout(); fig.subplots_adjust(wspace=0.12)
save(fig, "figD_local_energy_18e.png")

# ---- Figure E (report Figure 4): whole iteration = local energy + everything else, 1024 walkers, one GPU.
# (local energy alone, whole iteration).  Ours before: 6 e from the 6-electron calibration run (0.087 s),
# 18 e from bench18.py (0.577 s).  Authors: 7.1 iterations per second in their 9-site training (6 e),
# 0.33 s from their 240-minus-40-iteration smoke test (18 e).  Local energies from pieces*.py.
full = {("6 electrons", "ours before\n(Hessian + det)"): (R["pieces_6e_ours_det_eloc_hessian"]["seconds_per_call"], 0.087),
        ("6 electrons", "ours after\n(folx + slogdet)"): (R["pieces_6e_ours_slogdet_eloc_folx"]["seconds_per_call"],
                                                          R["full_6e_attention_folx_slogdet"]["seconds_per_iteration"]),
        ("6 electrons", "authors"): (R["pieces_6e_authors_eloc_folx"]["seconds_per_call"], 1 / 7.1),
        ("18 electrons", "ours before\n(Hessian + det)"): (0.476, 0.577),
        ("18 electrons", "ours after\n(folx + slogdet)"): (0.1296, R["full_18e_attention_folx_slogdet_b1024"]["seconds_per_iteration"]),
        ("18 electrons", "authors"): (0.1305, 0.33)}
fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1), gridspec_kw=dict(wspace=0.35))
for ax, size in zip(axes, ["6 electrons", "18 electrons"]):
    labs = [k[1] for k in full if k[0] == size]
    for x, lab in enumerate(labs):
        el, tot = full[(size, lab)]
        col = ORANGE if lab == "authors" else (GREEN if "after" in lab else BLUE)
        ax.bar(x, el, color=col, width=0.6)
        ax.bar(x, tot - el, bottom=el, color=col, alpha=0.35, width=0.6)
        ax.text(x, tot * 1.02, f"{tot:.3f} s", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(range(3)); ax.set_xticklabels(labs, fontsize=7)
    ax.set_ylim(0, max(full[(size, l)][1] for l in labs) * 1.18)
    ax.set_title(size, loc="left")
axes[0].set_ylabel("seconds per training iteration")
axes[0].legend(handles=[Patch(color="#777777"), Patch(color="#777777", alpha=0.35)],
               labels=["local energy", "everything else\n(moves, KFAC, loop)"], fontsize=7, loc="upper left")
fig.tight_layout()
save(fig, "figE_iteration_split.png")
print("figures written to", HERE)
