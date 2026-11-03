"""
Stage 8 (our code) -- the gradient test: SlaterNet eps = 10, 18 electrons on 27 sites,
continued from iteration 33,545 with the authors' gradient (real) and the full one.

Run after the runs and their frozen-weight evaluations (CPU is enough), from
learn/stage08_own_code/analysis/18_electrons_27_sites:
    uv run --no-project --python ../../../../.venv/bin/python --with matplotlib --with pyblock python gradient_test_results.py

Reads   runs/stage08/18_electrons_27_sites/training/slaternet_18e_eps10/            the original run
        runs/stage08/18_electrons_27_sites/gradient_test/training/<branch>/        the two branches
        runs/stage08/18_electrons_27_sites/gradient_test/evaluation/<branch>[_burnin2000_seed1]/
        runs/stage08/18_electrons_27_sites/evaluation/slaternet_18e_eps10[_at_033545]/
Writes  runs/stage08/18_electrons_27_sites/gradient_test/results/frozen_energies.csv
        runs/stage08/18_electrons_27_sites/gradient_test/results/training_windows.csv
        runs/stage08/18_electrons_27_sites/gradient_test/results/training_at_original_checkpoints.csv
        runs/stage08/18_electrons_27_sites/gradient_test/results/comparison.json   (row and checkpoint identity)
        runs/stage08/18_electrons_27_sites/gradient_test/figures/gradient_test.png

On 2026-11-03 the full branch was continued from 76,070 to 150,000, the
original's end (`launch_gradient_test.sh extend`); the real branch, a replay
of the original, was not.  The script works before and after that: it uses
whatever the folders hold, and the evaluation folders <run>_at_<iterations>.

Settled?  Each evaluation's energy is averaged over its first and second half
(steps 0-2499, 2500-4999).  If the two halves differ by more than twice their
combined blocking error, the walkers were still settling: then
`launch_gradient_test.sh settle` evaluates both branches again with 10x the
burn-in and another seed, and this script uses those evaluations too.  The
four blocks 0-99, 100-999, 1000-2999 and 3000-4999 are listed as well, as in
../../../../runs/stage08/18_electrons_27_sites/README.md.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                     # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "6_electrons_9_sites"))
from analyse_runs import ENERGY_UNIT_MEV, blocked_mean   # noqa: E402

DATA = HERE.parents[3] / "runs" / "stage08" / "18_electrons_27_sites"
ORIGINAL = DATA / "training" / "slaternet_18e_eps10"
TEST = DATA / "gradient_test"
BRANCHES = {"real": "slaternet_18e_eps10_from033545_real", "full": "slaternet_18e_eps10_from033545_full"}
BRANCH_STEP, FINAL_STEP = 33545, 76070             # the branch point; the end of the real branch
END_STEP = 150000                                   # the original's end; the full branch continued to it (2026-11-03)
N = 18
TO_MEV = ENERGY_UNIT_MEV / N                        # code units (whole system) -> meV per electron
BLOCKS = [(0, 100), (100, 1000), (1000, 3000), (3000, 5000)]
WINDOW = 1000                                       # iterations per point of the training curve
COLOURS = {"original": "0.45", "real": "#2e8b57", "full": "#c2410c"}
NAMES = {"original": "original run (real gradient)", "real": "branch, real gradient (GPU 1), to 76,070",
         "full": "branch, full gradient (GPU 0)"}


def md5(path):
    return hashlib.md5(path.read_bytes()).hexdigest()


def training(folder):
    stats = pd.read_csv(folder / "train_stats.csv")
    assert stats["step"].is_unique and (np.diff(stats["step"]) == 1).all(), f"{folder}: one row per iteration"
    return stats


def windows(stats, start, stop):
    """Mean energy and mean Im E_loc (meV per electron) in windows of WINDOW iterations."""
    rows = []
    for a in range(start, stop, WINDOW):
        part = stats[(stats["step"] >= a) & (stats["step"] < a + WINDOW)]
        if len(part) == WINDOW:
            rows.append(dict(start=a, energy=part["energy"].mean() * TO_MEV,
                             energy_imag=part["energy_imag"].mean() * TO_MEV))
    return pd.DataFrame(rows, columns=["start", "energy", "energy_imag"])


def evaluation(folder, training_stats, label, gradient):
    """One frozen-weight evaluation: energy, error, blocks, halves, training before its checkpoint."""
    summary = json.loads((folder / "summary.json").read_text())
    energies = pd.read_csv(folder / "energies.csv")
    if not (summary["steps_done"] == summary["steps_requested"] == len(energies)
            and summary["sanity_check"]["passed"] and summary["n_electrons"] == N):
        sys.exit(f"{folder}: unfinished, or its sanity check did not pass")
    series = energies["energy"].to_numpy() * TO_MEV
    frozen, frozen_error = blocked_mean(series)
    row = dict(evaluation=folder.name, label=label, gradient=gradient, checkpoint=summary["checkpoint_step"],
               burn_in=summary["burn_in"], seed=summary["seed"], steps=len(series),
               E_frozen=frozen, E_frozen_error=frozen_error)
    for a, b in BLOCKS:
        row[f"E_steps_{a}_{b - 1}"], row[f"E_steps_{a}_{b - 1}_error"] = blocked_mean(series[a:b])
    first, second = np.array_split(series, 2)
    (m1, e1), (m2, e2) = blocked_mean(first), blocked_mean(second)
    row.update(E_first_half=m1, E_second_half=m2, halves_difference=m2 - m1,
               halves_difference_error=float(np.hypot(e1, e2)))
    row["settled"] = bool(abs(m2 - m1) <= 2 * np.hypot(e1, e2))
    before = training_stats[training_stats["step"] < row["checkpoint"]]["energy"].to_numpy()[-2000:] * TO_MEV
    row["E_training_last_2000"], row["E_training_last_2000_error"] = blocked_mean(before)
    row["checkpoint_md5"] = summary["checkpoint_md5"]
    return row, series


def main():
    original = training(ORIGINAL)
    branches = {g: training(TEST / "training" / name) for g, name in BRANCHES.items()}
    ends = {"original": int(original["step"].iloc[-1]) + 1}       # iterations done
    for g, stats in branches.items():
        ends[g] = int(stats["step"].iloc[-1]) + 1
        if ends[g] < FINAL_STEP:
            sys.exit(f"the {g} branch has not reached {FINAL_STEP}: last row {ends[g] - 1}")
        copied = stats[stats["step"] < BRANCH_STEP].drop(columns="seconds")
        same = original[original["step"] < BRANCH_STEP].drop(columns="seconds")
        assert copied.equals(same), f"{g}: the copied rows differ from the original's"

    # 1. Training: the branches against the original, iteration by iteration.
    columns = ["energy", "variance", "energy_imag", "pmove", "step_width", "learning_rate", "fisher_step_sq"]
    mine = original[(original["step"] >= BRANCH_STEP) & (original["step"] < FINAL_STEP)].set_index("step")[columns]
    report = {}
    for g, stats in branches.items():
        theirs = stats[(stats["step"] >= BRANCH_STEP) & (stats["step"] < FINAL_STEP)].set_index("step")[columns]
        identical = int((mine == theirs).all(axis=1).sum())
        report[g] = dict(rows=len(theirs), identical_rows=identical)
        print(f"{g}: {identical} of {len(theirs)} rows from {BRANCH_STEP} identical to the original's")
    final_original = ORIGINAL / f"checkpoint_{FINAL_STEP:06d}.npz"
    for g, name in BRANCHES.items():
        final = TEST / "training" / name / f"checkpoint_{FINAL_STEP:06d}.npz"
        report[g]["final_checkpoint_identical_to_original"] = md5(final) == md5(final_original)
        print(f"{g}: checkpoint_{FINAL_STEP} identical to the original's: "
              f"{report[g]['final_checkpoint_identical_to_original']}")

    curves = {"original": windows(original, 25000, ends["original"])}
    curves.update({g: windows(stats, 33000, ends[g]) for g, stats in branches.items()})
    table = curves["original"].rename(columns={"energy": "original_energy", "energy_imag": "original_energy_imag"})
    for g in BRANCHES:
        table = table.merge(curves[g].rename(columns={"energy": f"{g}_energy", "energy_imag": f"{g}_energy_imag"}),
                            on="start", how="left")
    table.insert(0, "electrons", N)
    table.insert(1, "sites", 27)
    table = table.rename(columns={"start": "first_iteration"})

    # Rise: first 2,000 iterations after the branch point against the last 2,000.
    for name, stats in [("original", original)] + list(branches.items()):
        early = stats[(stats["step"] >= BRANCH_STEP) & (stats["step"] < BRANCH_STEP + 2000)]["energy"] * TO_MEV
        late = stats[(stats["step"] >= FINAL_STEP - 2000) & (stats["step"] < FINAL_STEP)]["energy"] * TO_MEV
        (e0, s0), (e1, s1) = blocked_mean(early), blocked_mean(late)
        print(f"{name:8s} training {BRANCH_STEP}-{BRANCH_STEP + 1999}: {e0:.4f}({s0:.4f})   "
              f"{FINAL_STEP - 2000}-{FINAL_STEP - 1}: {e1:.4f}({s1:.4f})   change {e1 - e0:+.4f}")
        if ends[name] >= END_STEP:
            end = stats[(stats["step"] >= END_STEP - 2000) & (stats["step"] < END_STEP)]["energy"] * TO_MEV
            e2, s2 = blocked_mean(end)
            print(f"{'':8s} training {END_STEP - 2000}-{END_STEP - 1}: {e2:.4f}({s2:.4f})   "
                  f"change since {FINAL_STEP - 2000}-{FINAL_STEP - 1}: {e2 - e1:+.4f}")

    # At the original's checkpoints after the branch point: the 2,000 training iterations before each.
    at_checkpoints = []
    for path in sorted(ORIGINAL.glob("checkpoint_*.npz")):
        step = int(path.stem.split("_")[1])
        if not BRANCH_STEP < step <= max(ends.values()):
            continue                     # (a branch that has not got that far gets nan there)
        row = dict(electrons=N, sites=27, original_checkpoint=step)
        for name, stats in [("original", original)] + list(branches.items()):
            part = stats[(stats["step"] >= step - 2000) & (stats["step"] < step)]["energy"] * TO_MEV
            row[f"{name}_training_last_2000"], row[f"{name}_training_last_2000_error"] = blocked_mean(part)
        row["full_minus_original"] = row["full_training_last_2000"] - row["original_training_last_2000"]
        at_checkpoints.append(row)
    at_checkpoints = pd.DataFrame(at_checkpoints)
    with pd.option_context("display.width", 250, "display.precision", 4):
        print(at_checkpoints.drop(columns=["electrons", "sites"]).to_string(index=False))

    # 2. Frozen weights.
    rows, series = [], {}
    for label, folder, stats, gradient in [
            ("original", DATA / "evaluation" / f"slaternet_18e_eps10_at_{BRANCH_STEP:06d}", original, "real"),
            ("original", DATA / "evaluation" / "slaternet_18e_eps10", original, "real")]:
        row, _ = evaluation(folder, stats, label, gradient)
        rows.append(row)
    for g, name in BRANCHES.items():
        for step in (FINAL_STEP, END_STEP):
            for suffix in ("", "_burnin2000_seed1"):
                folder = TEST / "evaluation" / f"{name}_at_{step:06d}{suffix}"
                if (folder / "summary.json").exists():
                    row, s = evaluation(folder, branches[g], g, g)
                    rows.append(row)
                    series[folder.name] = (g, step, bool(suffix), s)
    frozen = pd.DataFrame(rows)
    frozen.insert(1, "electrons", N)
    frozen.insert(2, "sites", 27)

    out_results, out_figures = TEST / "results", TEST / "figures"
    out_results.mkdir(exist_ok=True)
    out_figures.mkdir(exist_ok=True)
    frozen.to_csv(out_results / "frozen_energies.csv", index=False)
    table.to_csv(out_results / "training_windows.csv", index=False)
    at_checkpoints.to_csv(out_results / "training_at_original_checkpoints.csv", index=False)
    with pd.option_context("display.width", 250, "display.precision", 5):
        print(frozen.drop(columns=["checkpoint_md5"]).T.to_string())
    print("settled:", dict(zip(frozen["evaluation"], frozen["settled"])))
    (out_results / "comparison.json").write_text(json.dumps(report, indent=2))

    # 3. The figure.
    with plt.rc_context({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7,
                         "xtick.labelsize": 7.5, "ytick.labelsize": 7.5}):
        figure(curves, frozen, series, out_figures / "gradient_test.png")
    print(f"written: {out_results}, {out_figures / 'gradient_test.png'}")


def figure(curves, frozen, series, path):
    fig, (left, right) = plt.subplots(1, 2, figsize=(7.2, 3.1), constrained_layout=True,
                                      gridspec_kw=dict(width_ratios=[1.7, 1]))
    for name in ("original", "full", "real"):
        c = curves[name]
        style = dict(color=COLOURS[name], lw=1.4 if name == "original" else 1.1,
                     ls=(0, (3, 2)) if name == "real" else "-")
        left.plot((c["start"] + WINDOW / 2) / 1000, c["energy"], label=NAMES[name], **style)
    left.axvline(BRANCH_STEP / 1000, color="0.6", lw=0.6, ls=":")
    left.text(BRANCH_STEP / 1000 + 0.5, 0.97, "branch point 33,545", transform=left.get_xaxis_transform(),
              va="top", fontsize=7, color="0.35")
    for _, r in frozen.iterrows():
        if r["evaluation"].endswith("_burnin2000_seed1"):
            continue
        marker = "s" if r["label"] == "original" else "o"
        left.errorbar(r["checkpoint"] / 1000, r["E_frozen"], yerr=r["E_frozen_error"], fmt=marker, ms=4.5,
                      color=COLOURS[r["label"]], mec="black", mew=0.5, zorder=5)
        early = r["checkpoint"] < 50000                   # label to the right of early symbols, else left
        left.annotate(f"frozen {r['E_frozen']:.3f}".replace("-", "\u2212"), (r["checkpoint"] / 1000, r["E_frozen"]),
                      xytext=(5 if early else -5, 0), textcoords="offset points", ha="left" if early else "right",
                      va="center", fontsize=6.5, color=COLOURS[r["label"]])
    left.set_xlabel("iteration (thousands)")
    left.set_ylabel("E/N (meV)")
    left.set_xlim(25, max(c["start"].max() for c in curves.values()) / 1000 + 4)
    left.set_title("a   training (mean of 1,000 iterations); symbols: frozen weights", loc="left")
    left.legend(frameon=False, loc="center right")

    for name, (g, step, second, s) in series.items():
        whole = np.mean(s)
        centres, means, errors = [], [], []
        for a, b in BLOCKS:
            m, e = blocked_mean(s[a:b])
            centres.append((a + b) / 2)
            means.append((m - whole) * 1000)
            errors.append(e * 1000)
        marker = "o" if step == FINAL_STEP else "^"
        right.errorbar(np.array(centres) * (1.04 if second else 1.0), means, yerr=errors,
                       fmt=marker + "-", mfc="white" if second else COLOURS[g], color=COLOURS[g],
                       ms=4, lw=0.9, label=f"{g} at {step:,}{', burn-in 2000, seed 1' if second else ''}")
    right.axhline(0, color="0.6", lw=0.6)
    right.set_xscale("log")
    right.set_xlabel("evaluation step (blocks 0-99, 100-999,\n1000-2999, 3000-4999)")
    right.set_ylabel("block mean \u2212 mean (10$^{-3}$ meV)")
    right.set_title("b   have the evaluations settled?", loc="left")
    fig.suptitle("SlaterNet, \u03b5 = 10, 18 electrons on 27 sites, continued from iteration 33,545 "
                 "with the authors' (real) and the full gradient", x=0.01, ha="left", fontsize=8)
    right.legend(frameon=False, loc="lower right", fontsize=6.5)
    fig.savefig(path, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
