"""
Stage 8 (our code) -- Table I and Fig. 6 of the paper: 18 electrons on 27 sites.

Run after the frozen-weight evaluations of the Table I runs (CPU is enough),
from learn/stage08_own_code/analysis/18_electrons_27_sites:
    uv run --no-project --python ../../../../.venv/bin/python --with matplotlib --with pyblock python table1_results.py
    (add --gradient full for runs/stage08/18_electrons_27_sites_full_gradient/)

Reads   runs/stage08/18_electrons_27_sites_real_gradient/training/<run>/train_stats.csv         (training)
        runs/stage08/18_electrons_27_sites_real_gradient/evaluation/<run>[_at_<N>]/      (frozen weights, evaluate_weights.py)
Writes  runs/stage08/18_electrons_27_sites_real_gradient/results/table1.csv              energies against the paper's Table I
        runs/stage08/18_electrons_27_sites_real_gradient/results/structure.csv           where the electrons are
        runs/stage08/18_electrons_27_sites_real_gradient/figures/1_table1_energies.png
        runs/stage08/18_electrons_27_sites_real_gradient/figures/2_fig6_attention.png    paper Fig. 6: density and pair correlation
        runs/stage08/18_electrons_27_sites_real_gradient/figures/3_fig6_slaternet.png    the same for SlaterNet
        runs/stage08/18_electrons_27_sites_real_gradient/figures/4_sublattices_during_evaluation.png

What is compared.  The paper's Table I gives one energy per network and eps,
in meV per electron.  Ours is the frozen-weight energy: the final weights
held fixed, 5000 steps x 4096 walkers, error bar from blocking.  The
training average over the last 20 % is listed beside it; the two estimates
belong to different wave functions (evaluate_weights.py explains why) and
need not agree within their error bars.

The paper's error bars.  The captions of the paper's Figs. 3(b), 5 and 7
define its error bars as the standard deviation of the batch-averaged local
energy of one optimisation step, and Appendix C says they shrink as
1/sqrt(batch size).  Table I's caption does not say, but its errors
(0.007-0.03 meV) match that spread for 4096 walkers (our column
per_step_spread) -- not the error of a long average, which is about a
hundred times smaller.  So the paper's values are most likely averages over
training steps with that spread as the error bar.  It does not say over
which steps, so both of our estimates are listed and neither is taken as
the definitive comparison.  frozen_minus_paper_in_errors divides by the
paper's error as printed.

Where the electrons are.  At nu = 2/3 the crystal of the paper is a
honeycomb: of the three sublattices of the triangular moire lattice (sites
with n1 + n2 = 0, 1, 2 mod 3, each 9 of the 27) one is empty.  So the
fraction of electrons on each sublattice tells crystal from liquid: 1/3 each
for a liquid -- or for a crystal that wanders between its three
orientations -- and (0, 1/2, 1/2) for a crystal that stays put.  S(kappa)
(above 1 for a honeycomb; 4.5 for a perfect one, test_observables.py) and
the pair correlation see the crystal in both cases.
"""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ["JAX_PLATFORMS"] = "cpu"         # observables.py imports JAX; plotting needs no GPU

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                     # noqa: E402
from matplotlib.colors import TwoSlopeNorm           # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))                  # observables.py, shared by both systems
# From the 6-electron analysis, unchanged: the energy unit, the blocking error bar, the colours.
sys.path.insert(0, str(HERE.parent / "6_electrons_9_sites"))
import observables                                   # noqa: E402
from analyse_runs import ENERGY_UNIT_MEV, COLOURS, LABELS, blocked_mean   # noqa: E402

DATA = HERE.parents[3] / "runs" / "stage08" / "18_electrons_27_sites_real_gradient"
TRAINING = DATA / "training"
EVALUATION, FIGURES, RESULTS = DATA / "evaluation", DATA / "figures", DATA / "results"
GRADIENT = "real"


def use(gradient):
    """Point DATA and its subfolders at the runs of one gradient convention (--gradient)."""
    global DATA, TRAINING, EVALUATION, FIGURES, RESULTS, GRADIENT
    GRADIENT = gradient
    DATA = HERE.parents[3] / "runs" / "stage08" / f"18_electrons_27_sites_{gradient}_gradient"
    TRAINING = DATA / "training"
    EVALUATION, FIGURES, RESULTS = DATA / "evaluation", DATA / "figures", DATA / "results"
N = 18
FINAL_FRACTION = 0.2
KAPPA = 4 * np.pi / 3
# The paper's Table I (p. 7), meV per electron: (value, error); BP-ED has no error bar.
PAPER = {("attention", 10.0): (-32.070, 0.007), ("slaternet", 10.0): (-31.35, 0.02),
         ("attention", 5.0): (-59.127, 0.009), ("slaternet", 5.0): (-58.01, 0.03)}
BP_ED = {10.0: -31.32443, 5.0: -57.80848}
# Evaluation folders, in the order of the tables.  The last is SlaterNet eps = 10 at
# iteration 33,545, near its lowest training energy (the energy rose afterwards).
FOLDERS = ["attention_18e_eps10", "slaternet_18e_eps10", "attention_18e_eps5", "slaternet_18e_eps5",
           "slaternet_18e_eps10_at_033545"]

plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
                     "legend.fontsize": 7, "xtick.labelsize": 6, "ytick.labelsize": 6,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titlelocation": "left"})


# ------------------------------------------------------------------ loading

def sublattice_of_sites(supercell):
    """(n1 + n2) mod 3 of every moire site, in the order of observables.moire_sites."""
    sites = observables.moire_sites(supercell)
    n = np.linalg.solve(observables.MOIRE_LATTICE, sites.T)
    assert np.allclose(n, np.round(n), atol=1e-9), "sites must be lattice points"
    return np.round(n).astype(int).sum(axis=0) % 3


def load(folder_name):
    """Everything about one evaluation, or None if it is not there (yet)."""
    folder = EVALUATION / folder_name
    if not (folder / "summary.json").exists():
        return None
    summary = json.loads((folder / "summary.json").read_text())
    energies = pd.read_csv(folder / "energies.csv")
    totals = dict(np.load(folder / "observables.npz"))
    steps = int(totals.pop("steps_done"))
    if not steps == summary["steps_done"] == summary["steps_requested"] == len(energies):
        sys.exit(f"{folder}: unfinished, or its files disagree about the number of steps")
    if summary["n_electrons"] != N or not summary["sanity_check"]["passed"]:
        sys.exit(f"{folder}: not an 18-electron evaluation that passed its sanity check")
    if summary["gradient"] != GRADIENT:                 # never mix the two conventions in one table
        sys.exit(f"{folder}: evaluated a run with --gradient {summary['gradient']}, not {GRADIENT}")
    n_configurations = steps * summary["batch_size"]
    assert totals["density"].sum() == N * n_configurations, "every electron counted once"
    supercell = np.array(summary["supercell"])
    to_meV = ENERGY_UNIT_MEV / N                         # code units (whole system) -> meV per electron

    run = dict(folder=folder_name, network=summary["network"], eps=float(summary["eps"]),
               step=summary["checkpoint_step"], final=summary["final_checkpoint"], summary=summary,
               totals=totals, supercell=supercell, n_configurations=n_configurations)
    run["series"] = energies["energy"].to_numpy() * to_meV
    run["frozen"], run["frozen_error"] = blocked_mean(run["series"])
    for part in ("kinetic", "moire", "coulomb"):
        run[part] = energies[part].mean() * to_meV
    run["sigma_local_energy_meV"] = np.sqrt(energies["variance"].mean()) * ENERGY_UNIT_MEV
    run["pmove"] = energies["pmove"].mean()

    # Training: the last 20 % of the iterations before this checkpoint.
    training = pd.read_csv(TRAINING / folder_name.split("_at_")[0] / "train_stats.csv", usecols=["step", "energy"])
    before = training[training["step"] < run["step"]]["energy"].to_numpy() * to_meV
    run["training_series"] = before[int(len(before) * (1 - FINAL_FRACTION)):]
    run["training"], run["training_error"] = blocked_mean(run["training_series"])

    # Where the electrons are.
    run["occupation"] = totals["occupation"] / n_configurations
    sublattice = sublattice_of_sites(supercell)
    occupations = energies[[f"occupation_{k}" for k in range(len(sublattice))]].to_numpy()
    run["sublattice_series"] = np.stack([occupations[:, sublattice == c].sum(axis=1) / N for c in range(3)], 1)
    run["sublattice"] = run["sublattice_series"].mean(axis=0)
    q = np.linalg.norm(np.array(summary["q_vectors"]), axis=1)
    S, S_connected = observables.structure_factor(totals["rho_q"], totals["rho_q_squared"], n_configurations, N)
    at_kappa = np.isclose(q, KAPPA, atol=1e-6)
    assert at_kappa.sum() == 6, "expected the 6 corners of the moire Brillouin zone"
    run["S_kappa"], run["S_kappa_connected"] = S[at_kappa].mean(), S_connected[at_kappa].mean()
    halves = np.array_split(energies["S_kappa"].to_numpy(), 2)
    run["S_kappa_halves"] = (halves[0].mean(), halves[1].mean())
    return run


def label_for(run):
    text = f"{LABELS[run['network']]}, \u03b5 = {run['eps']:g}"
    return text if run["final"] else f"{text}, at iteration {run['step']:,}"


# ------------------------------------------------------------------ tables

def energy_table(runs):
    rows = []
    for r in runs:
        paper, paper_error = PAPER[(r["network"], r["eps"])]
        difference_error = np.hypot(r["frozen_error"], paper_error)
        rows.append(dict(
            run=r["folder"], electrons=N, sites=r["summary"]["n_sites"], network=LABELS[r["network"]],
            eps=r["eps"], checkpoint=r["step"],
            E_frozen=r["frozen"], E_frozen_error=r["frozen_error"],
            E_training_last20pct=r["training"], E_training_last20pct_error=r["training_error"],
            E_paper_table1=paper, E_paper_table1_error=paper_error,
            frozen_minus_paper=r["frozen"] - paper, frozen_minus_paper_error=difference_error,
            frozen_minus_paper_in_errors=(r["frozen"] - paper) / difference_error,
            E_bp_ed=BP_ED[r["eps"]], frozen_minus_bp_ed=r["frozen"] - BP_ED[r["eps"]],
            # Spread of ONE step's batch-mean energy (4096 walkers), not an error of the mean:
            # compare it with the size of the paper's error bars.
            per_step_spread=np.std(r["series"], ddof=1),
            training_minus_paper=r["training"] - paper,
            training_per_step_spread=np.std(r["training_series"], ddof=1),
            kinetic=r["kinetic"], moire=r["moire"], coulomb=r["coulomb"],
            sigma_local_energy_meV=r["sigma_local_energy_meV"], pmove=r["pmove"],
            sanity_first_200_steps_meV=r["summary"]["sanity_check"]["evaluation_first_steps"] * ENERGY_UNIT_MEV / N,
            configurations=r["n_configurations"], steps_per_second=r["summary"]["steps_per_second"]))
    return pd.DataFrame(rows)


def correlation_energies(runs):
    """E(self-attention) - E(SlaterNet) at each eps, final weights: ours and the paper's."""
    final = {(r["network"], r["eps"]): r for r in runs if r["final"]}
    rows = []
    for eps in (10.0, 5.0):
        if ("attention", eps) in final and ("slaternet", eps) in final:
            a, s = final[("attention", eps)], final[("slaternet", eps)]
            rows.append(dict(electrons=N, sites=a["summary"]["n_sites"], eps=eps, ours=a["frozen"] - s["frozen"],
                             ours_error=np.hypot(a["frozen_error"], s["frozen_error"]),
                             paper=PAPER[("attention", eps)][0] - PAPER[("slaternet", eps)][0]))
    return pd.DataFrame(rows)


def structure_table(runs):
    return pd.DataFrame([dict(
        run=r["folder"], electrons=N, sites=r["summary"]["n_sites"], network=LABELS[r["network"]],
            eps=r["eps"], checkpoint=r["step"],
        S_kappa=r["S_kappa"], S_kappa_connected=r["S_kappa_connected"],
        S_kappa_first_half=r["S_kappa_halves"][0], S_kappa_second_half=r["S_kappa_halves"][1],
        sublattice_0=r["sublattice"][0], sublattice_1=r["sublattice"][1], sublattice_2=r["sublattice"][2],
        site_occupation_min=r["occupation"].min(), site_occupation_max=r["occupation"].max())
        for r in runs])


# ------------------------------------------------------------------ figures

def energy_figure(runs, path):
    """Frozen-weight and training energies minus the paper's Table I value."""
    fig, ax = plt.subplots(figsize=(5.2, 2.6), constrained_layout=True)
    ax.axvline(0, color="0.5", lw=0.6)
    for y, r in enumerate(runs):
        colour = COLOURS[r["network"]]
        paper, paper_error = PAPER[(r["network"], r["eps"])]
        ax.fill_betweenx([y - 0.35, y + 0.35], -paper_error, paper_error, color="0.88", lw=0)
        ax.errorbar(r["training"] - paper, y + 0.12, xerr=r["training_error"], fmt="o", mfc="white",
                    mec=colour, ecolor=colour, ms=4, label="training, last 20 %" if y == 0 else None)
        ax.errorbar(r["frozen"] - paper, y - 0.12, xerr=r["frozen_error"], fmt="o", color=colour, ms=4,
                    label="frozen weights" if y == 0 else None)
    ax.set_yticks(range(len(runs)), [label_for(r) for r in runs])
    ax.set_ylim(len(runs) - 0.5, -0.5)
    ax.set_xlabel("E/N \u2212 paper's Table I (meV)      (grey: the paper's error bar)")
    ax.set_title("Our 27-site energies relative to Table I (right = higher energy)"
                 + ("" if GRADIENT == "real" else ", full gradient"))
    ax.legend(frameon=False, loc="lower right")
    fig.savefig(path, dpi=300)
    plt.close(fig)


def tiled_mesh(supercell, bins, centred):
    """Bin corners of the cell and its 8 neighbours (Cartesian), for a periodic picture."""
    edges = np.linspace(0.0, 1.0, bins + 1) - (0.5 if centred else 0.0)
    s1, s2 = np.meshgrid(edges, edges, indexing="ij")
    for i in (-1, 0, 1):
        for j in (-1, 0, 1):
            yield (supercell[0, 0] * (s1 + i) + supercell[0, 1] * (s2 + j),
                   supercell[1, 0] * (s1 + i) + supercell[1, 1] * (s2 + j))


def draw_cell(ax, supercell, centred):
    corners = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]) - (0.5 if centred else 0.0)
    xy = corners @ supercell.T
    ax.plot(xy[:, 0], xy[:, 1], "--", color="0.85" if not centred else "0.3", lw=0.6)
    return xy


def fig6_figure(runs, network, path):
    """The paper's Fig. 6 layout: density (top) and pair correlation (bottom), eps = 10 left, 5 right."""
    by_eps = {r["eps"]: r for r in runs if r["network"] == network and r["final"]}
    fig, axes = plt.subplots(2, 2, figsize=(6.4, 6.0), constrained_layout=True)
    densities = {eps: observables.density_map(r["totals"]["density"], r["n_configurations"], r["supercell"])
                 for eps, r in by_eps.items()}
    pairs = {eps: observables.pair_correlation_map(r["totals"]["pairs"], r["n_configurations"], N, r["supercell"])
             for eps, r in by_eps.items()}
    for column, eps in enumerate((10.0, 5.0)):
        r = by_eps.get(eps)
        if r is None:
            axes[0, column].set_axis_off(), axes[1, column].set_axis_off()
            continue
        bins = r["summary"]["density_bins"]
        for row, (values, centred) in enumerate(((densities[eps], False), (pairs[eps], True))):
            ax = axes[row, column]
            if row == 0:
                style = dict(cmap="magma", vmin=0, vmax=max(m.max() for m in densities.values()))
            else:
                style = dict(cmap="RdBu_r", norm=TwoSlopeNorm(vmin=0.0, vcenter=1.0,
                                                              vmax=max(m.max() for m in pairs.values())))
            for X, Y in tiled_mesh(r["supercell"], bins, centred):
                image = ax.pcolormesh(X, Y, values, shading="flat", rasterized=True, **style)
            xy = draw_cell(ax, r["supercell"], centred)
            margin = 0.6
            ax.set_xlim(xy[:, 0].min() - margin, xy[:, 0].max() + margin)
            ax.set_ylim(xy[:, 1].min() - margin, xy[:, 1].max() + margin)
            ax.set_aspect("equal")
            if centred:
                ax.plot(0, 0, "+", color="k", ms=5, mew=0.8)       # the reference electron
            what = "density n(r)" if row == 0 else "pair correlation g(\u0394r)"
            ax.set_title(f"{'abcd'[2 * row + column]}   {what}, \u03b5 = {eps:g}")
            ax.set_xlabel("\u0394x / a$_M$" if centred else "x / a$_M$")
            ax.set_ylabel("\u0394y / a$_M$" if centred else "y / a$_M$")
            if column == 1:
                fig.colorbar(image, ax=axes[row, :], shrink=0.8,
                             label="n(r)  (a$_M^{-2}$)" if row == 0 else "g(\u0394r)  (1 = uniform, independent electrons)")
    fig.suptitle(f"{LABELS[network]}, 27 sites, 18 electrons (dashed: the supercell) -- paper Fig. 6"
                 + ("" if GRADIENT == "real" else ", full gradient"),
                 x=0.02, ha="left", fontsize=8)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def sublattice_figure(runs, path, window=100):
    """Fraction of the electrons on each of the three sublattices, step by step."""
    fig, axes = plt.subplots(len(runs), 1, figsize=(6.4, 1.1 * len(runs) + 0.4), sharex=True,
                             constrained_layout=True)
    for ax, r in zip(np.atleast_1d(axes), runs):
        smooth = pd.DataFrame(r["sublattice_series"]).rolling(window).mean().to_numpy()
        for c, shade in zip(range(3), ("0.15", "0.45", "0.7")):
            ax.plot(smooth[:, c], color=shade, lw=0.8, label=f"sublattice {c}")
        ax.axhline(1 / 3, color=COLOURS[r["network"]], lw=0.8, ls="--")
        ax.set_ylim(0, 0.6)
        ax.set_title(label_for(r))
        ax.set_ylabel("fraction")
    np.atleast_1d(axes)[0].legend(frameon=False, ncol=3, loc="upper right")
    np.atleast_1d(axes)[-1].set_xlabel("evaluation step")
    fig.suptitle(f"Electrons on each sublattice during the evaluation ({window}-step moving average)\n"
                 "dashed: 1/3 each; a honeycomb that stays put would give 0, 1/2, 1/2", x=0.02, ha="left", fontsize=8)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--gradient", choices=["real", "full"], default="real",
                        help="real: runs/stage08/18_electrons_27_sites_real_gradient (the authors'); "
                             "full: runs/stage08/18_electrons_27_sites_full_gradient")
    use(parser.parse_args().gradient)
    runs = [r for r in (load(name) for name in FOLDERS) if r is not None]
    if not runs:
        sys.exit(f"no finished evaluations in {EVALUATION}")
    FIGURES.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    tables = {"table1.csv": energy_table(runs), "structure.csv": structure_table(runs),
              "correlation_energy.csv": correlation_energies(runs)}
    for name, table in tables.items():
        table.to_csv(RESULTS / name, index=False)
        with pd.option_context("display.width", 200, "display.precision", 4):
            print(f"\n{name}\n{table.T.to_string()}")
    energy_figure(runs, FIGURES / "1_table1_energies.png")
    fig6_figure(runs, "attention", FIGURES / "2_fig6_attention.png")
    fig6_figure(runs, "slaternet", FIGURES / "3_fig6_slaternet.png")
    sublattice_figure(runs, FIGURES / "4_sublattices_during_evaluation.png")
    print(f"\ntables written to {RESULTS}, figures to {FIGURES}")


if __name__ == "__main__":
    main()
