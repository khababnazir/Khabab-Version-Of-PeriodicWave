"""
Stage 8 -- figures and table from the frozen-weight evaluations.

Run after evaluate_weights.py (CPU is enough; matplotlib is layered on top of
the project environment without changing it):
    uv run --no-project --python ../../../../.venv/bin/python --with matplotlib python plot_weights.py
(from learn/archive/stage08_authors_code/analysis)

Reads runs/archive/stage08_authors_code/evaluation/<run>/ and runs/archive/stage08_authors_code/training/<run>/, writes
    runs/archive/stage08_authors_code/results/energies_meV_per_electron.csv   energies, one row per run
    runs/archive/stage08_authors_code/results/electron_structure.csv          where the electrons are, one row per run
    runs/archive/stage08_authors_code/figures/2_frozen_weight_energies.png    (a) frozen vs training vs Fig. 5, (b) blocking
    runs/archive/stage08_authors_code/figures/3_electron_density.png          n(r)                  -- paper Fig. 6(a, b)
    runs/archive/stage08_authors_code/figures/4_pair_correlation.png          g(dr) maps and g(r)   -- paper Fig. 6(c, d)
    runs/archive/stage08_authors_code/figures/5_site_filling_and_structure_factor.png
    runs/archive/stage08_authors_code/figures/6_site_occupation_during_evaluation.png
(Figure 1, the training curves, comes from analyse_runs.py.)

It refuses evaluation folders whose three files disagree about the number of
steps (a crash between two writes).  The titles describe what is plotted;
what the plots show is for the reader, not written into the code.

Comparing with Fig. 6.  The paper's Fig. 6 is for 27 sites (18 electrons);
our runs are the 9-site, 6-electron system of Fig. 5.  The comparison is
qualitative: does eps = 10 look like a liquid and eps = 5 like a crystal?
"""
import json
import os
import sys
from pathlib import Path

# observables.py imports JAX, which would otherwise start up a visible GPU and reserve
# most of its memory.  Plotting needs no GPU, and must not share one with a training run.
os.environ["JAX_PLATFORMS"] = "cpu"

import numpy as np
import pandas as pd
import pyblock
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                     # noqa: E402
from matplotlib.colors import TwoSlopeNorm           # noqa: E402
from matplotlib.patches import Patch                 # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import observables                                   # noqa: E402
from analyse_runs import ENERGY_UNIT_MEV, FIG5, COLOURS, LABELS, blocked_mean, describe  # noqa: E402

STAGE08 = HERE.parents[3] / "runs" / "archive" / "stage08_authors_code"
TRAINING, EVALUATION = STAGE08 / "training", STAGE08 / "evaluation"
FIGURES, RESULTS = STAGE08 / "figures", STAGE08 / "results"
ORDER = ["psiformer_eps10", "slaternet_eps10", "psiformer_eps5", "slaternet_eps5"]
FIG5_READING_ACCURACY = 0.05      # meV per electron, our by-eye reading of the plot
KAPPA = 4 * np.pi / 3             # corner of the moire Brillouin zone (a_M = 1)
G = 4 * np.pi / np.sqrt(3)        # moire reciprocal lattice vector

# Three font sizes only: labels, legends/annotations, ticks.
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
                     "legend.fontsize": 7, "xtick.labelsize": 6, "ytick.labelsize": 6,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titlelocation": "left"})


# ------------------------------------------------------------------ loading

def load(name):
    """Everything about one evaluated run, or None if it has not been evaluated."""
    folder = EVALUATION / name
    if not (folder / "summary.json").exists():
        return None
    summary = json.loads((folder / "summary.json").read_text())
    energies = pd.read_csv(folder / "energies.csv")
    totals = dict(np.load(folder / "observables.npz"))
    command_line = summary["authors_command_line"]
    N = summary["n_electrons"]
    to_meV_per_electron = ENERGY_UNIT_MEV / N
    supercell = np.array(summary["supercell"])

    # The three files are written one after the other; after a crash between
    # two writes they would describe different numbers of steps.
    steps = int(totals.pop("steps_done"))
    if not steps == summary["steps_done"] == len(energies):
        sys.exit(f"{folder}: the output files disagree (observables.npz {steps} steps, "
                 f"summary.json {summary['steps_done']}, energies.csv {len(energies)}); "
                 "evaluate again with --overwrite")
    n_configurations = steps * summary["batch_size"]
    assert totals["density"].sum() == N * n_configurations, "every electron counted once"
    assert totals["patterns"].sum() == n_configurations, "every configuration counted once"

    run = dict(name=name, network=command_line[-1], eps=round(1.0 / float(command_line[3]), 6),
               summary=summary, totals=totals, N=N, supercell=supercell,
               n_configurations=n_configurations)

    # Energies: per electron, in meV.  The error bar from blocking, as for the training.
    per_electron = energies["energy"].to_numpy() * to_meV_per_electron
    run["series"] = per_electron
    run["frozen"], run["frozen_error"] = blocked_mean(per_electron)
    for part in ("kinetic", "moire", "coulomb"):
        run[part] = energies[part].mean() * to_meV_per_electron
    run["sigma_local_energy_meV"] = np.sqrt(energies["variance"].mean()) * ENERGY_UNIT_MEV
    run["pmove"] = energies["pmove"].mean()
    training = describe(TRAINING / name)
    run["training"], run["training_error"] = training["mean"], training["error"]
    run["training_drift"], run["training_drift_error"] = training["drift"], training["drift_error"]
    run["fig5"] = FIG5[(run["eps"], run["network"])]

    # Which sites are filled.
    patterns = totals["patterns"] / n_configurations
    honeycombs = observables.honeycomb_codes(supercell)
    run["p_honeycombs"] = patterns[honeycombs]
    run["p_double"] = patterns[observables.SOME_SITE_DOUBLY_OCCUPIED]
    run["p_other"] = 1.0 - run["p_honeycombs"].sum() - run["p_double"]
    run["occupation"] = totals["occupation"] / n_configurations
    # If the one-electron-per-site configurations were random arrangements,
    # 3 of the 84 would be honeycombs.
    run["p_honeycombs_if_random"] = 3 / 84 * (1.0 - run["p_double"])

    # Site-pair correlation <n_k n_l> / (<n_k> <n_l>), averaged over the pairs
    # at one distance.  Dividing pair by pair removes the uneven site
    # occupations, so 1 means "independent sites".  Equal mixture of the three
    # honeycombs: 0.75 for neighbours, 1.5 at sqrt(3).  Random filling with one
    # electron per site: 15/16 for both.  But a SINGLE honeycomb that stays in
    # place gives 1 for its filled pairs (and 0/0 for empty sites), so these
    # numbers alone cannot rule out a crystal: read them with the filling
    # patterns and S(kappa).
    sites = observables.moire_sites(supercell)
    distance = np.array([[float(np.linalg.norm(observables.minimum_image(np.asarray(a - b), supercell)))
                          for b in sites] for a in sites])
    ratio = (totals["occupation_products"] / n_configurations) / np.outer(run["occupation"], run["occupation"])
    run["g_site_neighbours"] = ratio[np.isclose(distance, 1.0)].mean()
    run["g_site_sqrt3"] = ratio[np.isclose(distance, np.sqrt(3))].mean()

    # Is the spatial pattern stationary?  Per-step S(kappa) and site occupations.
    run["occupation_series"] = energies[[f"occupation_{k}" for k in range(observables.N_SITES)]].to_numpy()
    halves = np.array_split(energies["S_kappa"].to_numpy(), 2)
    run["S_kappa_halves"] = (halves[0].mean(), halves[1].mean())

    # Structure factor at the supercell's wave vectors.
    run["q"] = np.array(summary["q_vectors"])
    run["S"], run["S_connected"] = observables.structure_factor(
        totals["rho_q"], totals["rho_q_squared"], n_configurations, N)
    at_kappa = np.isclose(np.linalg.norm(run["q"], axis=1), KAPPA, atol=1e-6)
    assert at_kappa.sum() == 6, "expected the 6 corners of the moire Brillouin zone"
    run["S_kappa"], run["S_kappa_connected"] = run["S"][at_kappa].mean(), run["S_connected"][at_kappa].mean()
    return run


def energy_table(runs):
    """Energies in meV per electron (sigma_local_energy: meV, whole system), one row per run."""
    return pd.DataFrame([dict(
        run=r["name"], network=LABELS[r["network"]], eps=r["eps"],
        E_training_last20pct=r["training"], E_training_last20pct_error=r["training_error"],
        training_drift=r["training_drift"], training_drift_error=r["training_drift_error"],
        E_frozen_weights=r["frozen"], E_frozen_weights_error=r["frozen_error"],
        E_fig5_read_by_eye=r["fig5"],
        kinetic=r["kinetic"], moire=r["moire"], coulomb=r["coulomb"],
        sigma_local_energy_meV=r["sigma_local_energy_meV"], pmove=r["pmove"],
        evaluation_steps=r["summary"]["steps_done"], configurations=r["n_configurations"],
        steps_per_second=r["summary"]["steps_per_second"]) for r in runs])


def structure_table(runs):
    """Where the electrons are, one row per run (definitions in runs/archive/stage08_authors_code/results/README.md)."""
    return pd.DataFrame([dict(
        run=r["name"], network=LABELS[r["network"]], eps=r["eps"],
        P_honeycomb=r["p_honeycombs"].sum(), P_largest_honeycomb=r["p_honeycombs"].max(),
        P_honeycomb_if_random=r["p_honeycombs_if_random"], P_two_nearest_one_site=r["p_double"],
        g_site_neighbours=r["g_site_neighbours"], g_site_sqrt3=r["g_site_sqrt3"],
        occupation_min=r["occupation"].min(), occupation_max=r["occupation"].max(),
        S_kappa=r["S_kappa"], S_kappa_connected=r["S_kappa_connected"],
        S_kappa_first_half=r["S_kappa_halves"][0], S_kappa_second_half=r["S_kappa_halves"][1])
        for r in runs])


# ------------------------------------------------------------------ helpers

def cell_mesh(supercell, bins, centred=False):
    """Corners of the histogram bins in Cartesian coordinates, for pcolormesh."""
    edges = np.linspace(0.0, 1.0, bins + 1) - (0.5 if centred else 0.0)
    s1, s2 = np.meshgrid(edges, edges, indexing="ij")
    return supercell[0, 0] * s1 + supercell[0, 1] * s2, supercell[1, 0] * s1 + supercell[1, 1] * s2


def draw_cell_outline(ax, supercell, centred=False):
    corners = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]) - (0.5 if centred else 0.0)
    xy = corners @ supercell.T
    ax.plot(xy[:, 0], xy[:, 1], "--", color="0.3", lw=0.6)


def label_for(run):
    return f"{LABELS[run['network']]}, \u03b5 = {run['eps']:g}"


def line_style(run):
    return dict(color=COLOURS[run["network"]], ls="-" if run["eps"] == 5.0 else "--")


# ------------------------------------------------------------------ figures

def energy_figure(runs, path):
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.2, 2.8), constrained_layout=True,
                                     gridspec_kw=dict(width_ratios=[1.1, 1]))

    # (a) everything relative to Fig. 5, which we only know to +-0.05 meV.
    ax_a.axvspan(-FIG5_READING_ACCURACY, FIG5_READING_ACCURACY, color="0.92", lw=0)
    ax_a.axvline(0, color="0.6", lw=0.6)
    for y, r in enumerate(runs):
        colour = COLOURS[r["network"]]
        ax_a.errorbar(r["training"] - r["fig5"], y + 0.12, xerr=r["training_error"], fmt="o",
                      mfc="white", mec=colour, ecolor=colour, ms=4, label="training, last 20%" if y == 0 else None)
        ax_a.errorbar(r["frozen"] - r["fig5"], y - 0.12, xerr=r["frozen_error"], fmt="o",
                      color=colour, ms=4, label="frozen weights" if y == 0 else None)
    ax_a.set_yticks(range(len(runs)), [label_for(r) for r in runs])
    ax_a.set_ylim(len(runs) - 0.5, -0.5)              # first run at the top
    ax_a.set_xlabel("E/N \u2212 Fig. 5 value (meV)")
    ax_a.set_title("a   Frozen-weight and training energy estimates,\n     relative to Fig. 5")
    handles, labels = ax_a.get_legend_handles_labels()
    handles.append(Patch(facecolor="0.92", edgecolor="0.6", lw=0.5))
    labels.append("Fig. 5 read by eye: \u00b10.05 meV")
    ax_a.legend(handles, labels, frameon=False, loc="upper center", bbox_to_anchor=(0.45, -0.2),
                ncol=3, handletextpad=0.4, columnspacing=1.0, fontsize=6)

    # (b) blocking: the error bar levels off once blocks outlast the correlations.
    for r in runs:
        stats = pyblock.blocking.reblock(r["series"])
        sizes = np.array([2 ** s.block for s in stats])
        errors = np.array([s.std_err for s in stats])
        keep = np.array([s.ndata for s in stats]) >= 16
        ax_b.plot(sizes[keep], errors[keep] * 1e3, marker=".", ms=3, lw=0.9, label=label_for(r),
                  **line_style(r))
        optimal = pyblock.blocking.find_optimal_block(len(r["series"]), stats)[0]
        if not np.isnan(optimal):
            ax_b.plot(sizes[int(optimal)], errors[int(optimal)] * 1e3, "o", mfc="none",
                      mec=COLOURS[r["network"]], ms=6)
    ax_b.set_xscale("log", base=2)
    ax_b.set_ylim(bottom=0)
    ax_b.set_xlabel("block length (evaluation steps)")
    ax_b.set_ylabel("standard error of E/N (\u03bceV)")
    ax_b.set_title("b   Blocking analysis of the frozen-weight\n     energies (circle: block length used)")
    ax_b.legend(frameon=False, fontsize=6, loc="lower right", bbox_to_anchor=(1.0, 0.17))
    fig.savefig(path, dpi=300)
    plt.close(fig)


def density_figure(runs, path):
    by_key = {(r["network"], r["eps"]): r for r in runs}
    maps = {key: observables.density_map(r["totals"]["density"], r["n_configurations"], r["supercell"])
            for key, r in by_key.items()}
    top = max(m.max() for m in maps.values())
    fig, axes = plt.subplots(2, 2, figsize=(6.4, 3.7), constrained_layout=True)
    for row, network in enumerate(("CustomPsiformer", "SlaterNet")):
        for column, eps in enumerate((10.0, 5.0)):
            ax = axes[row, column]
            r = by_key.get((network, eps))
            if r is None:
                ax.set_axis_off()
                continue
            X, Y = cell_mesh(r["supercell"], r["summary"]["density_bins"])
            image = ax.pcolormesh(X, Y, maps[(network, eps)], cmap="magma", vmin=0, vmax=top,
                                  shading="flat", rasterized=True)
            draw_cell_outline(ax, r["supercell"])
            ax.set_aspect("equal")
            ax.set_title(label_for(r))
            if row == 1:
                ax.set_xlabel("x / a$_M$")
            if column == 0:
                ax.set_ylabel("y / a$_M$")
    fig.colorbar(image, ax=axes, shrink=0.8, label="electron density n(r)  (a$_M^{-2}$)")
    fig.suptitle("Where the electrons are: density in the 3 \u00d7 3 supercell (paper Fig. 6a, b: 27 sites)",
                 x=0.02, ha="left", fontsize=8)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def pair_correlation_figure(runs, path):
    by_key = {(r["network"], r["eps"]): r for r in runs}
    maps = {key: observables.pair_correlation_map(r["totals"]["pairs"], r["n_configurations"], r["N"],
                                                   r["supercell"]) for key, r in by_key.items()}
    top = max(m.max() for m in maps.values())
    norm = TwoSlopeNorm(vmin=0.0, vcenter=1.0, vmax=top)       # white = uncorrelated (g = 1)

    fig = plt.figure(figsize=(7.2, 4.0), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.05])
    for row, network in enumerate(("CustomPsiformer", "SlaterNet")):
        for column, eps in enumerate((10.0, 5.0)):
            r = by_key.get((network, eps))
            ax = fig.add_subplot(grid[row, column])
            if r is None:
                ax.set_axis_off()
                continue
            X, Y = cell_mesh(r["supercell"], r["summary"]["density_bins"], centred=True)
            image = ax.pcolormesh(X, Y, maps[(network, eps)], cmap="RdBu_r", norm=norm,
                                  shading="flat", rasterized=True)
            draw_cell_outline(ax, r["supercell"], centred=True)
            ax.plot(0, 0, "+", color="k", ms=5, mew=0.8)        # the reference electron
            ax.set_aspect("equal")
            ax.set_title(label_for(r))
            if row == 1:
                ax.set_xlabel("\u0394x / a$_M$")
            if column == 0:
                ax.set_ylabel("\u0394y / a$_M$")
    fig.colorbar(image, ax=fig.axes, location="left", shrink=0.7, pad=0.02,
                 label="g(\u0394r), relative to uniform independent electrons")

    # g(r) averaged over directions, one panel per eps.
    for row, eps in enumerate((10.0, 5.0)):
        ax = fig.add_subplot(grid[row, 2])
        for r in [r for r in runs if r["eps"] == eps]:
            distance, g_r = observables.radial_pair_correlation(
                r["totals"]["pair_distances"], r["n_configurations"], r["N"], r["supercell"],
                r["summary"]["r_max"])
            ax.plot(distance, g_r, color=COLOURS[r["network"]], lw=1.0, label=LABELS[r["network"]])
        ax.axvline(1.0, color="0.6", lw=0.5, ls=":")           # nearest-neighbour site distance
        ax.axhline(1.0, color="0.6", lw=0.5)
        ax.set_title(f"g(r), \u03b5 = {eps:g}   (dotted: site spacing)")
        ax.set_ylabel("g(r)")
        if row == 1:
            ax.set_xlabel("distance r / a$_M$")
        ax.legend(frameon=False, loc="lower right")
        ax.set_xlim(0, runs[0]["summary"]["r_max"])
    fig.suptitle("Where the other electrons are, seen from one electron (paper Fig. 6c, d: 27 sites)",
                 x=0.02, ha="left", fontsize=8)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def order_figure(runs, path):
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.2, 2.8), constrained_layout=True,
                                     gridspec_kw=dict(width_ratios=[1.15, 1]))

    # (a) probability of each kind of filling pattern.
    honeycomb_shades = ["#5e3c99", "#8e6bbf", "#c2a5dd"]
    for y, r in enumerate(runs):
        left = 0.0
        for k, p in enumerate(r["p_honeycombs"]):
            ax_a.barh(y, p, left=left, color=honeycomb_shades[k], height=0.6,
                      label=f"honeycomb {k + 1}" if y == 0 else None)
            left += p
        ax_a.barh(y, r["p_other"], left=left, color="0.75", height=0.6,
                  label="other, one electron per site" if y == 0 else None)
        left += r["p_other"]
        ax_a.barh(y, r["p_double"], left=left, color="white", edgecolor="0.4", lw=0.5, height=0.6,
                  label="two electrons nearest one site" if y == 0 else None)
        ax_a.plot([r["p_honeycombs_if_random"]] * 2, [y - 0.38, y + 0.38], color="k", lw=0.9,
                  label="honeycombs if the filling were random" if y == 0 else None)
    ax_a.set_yticks(range(len(runs)), [label_for(r) for r in runs])
    ax_a.set_ylim(len(runs) - 0.5, -0.5)              # first run at the top
    ax_a.set_xlim(0, 1)
    ax_a.set_xlabel("fraction of sampled configurations")
    ax_a.set_title("a   Probabilities of site-filling patterns")
    ax_a.legend(frameon=False, fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=3)

    # (b) structure factor at every wave vector of the supercell.  The full S(q),
    # not the connected part: if the sampling has settled into one honeycomb,
    # its order sits in |<rho_q>|^2, which the connected part removes.  At g the
    # peak is the moire lattice itself (every electron near a site).
    for r in runs:
        q = np.linalg.norm(r["q"], axis=1)
        style = line_style(r)
        ax_b.plot(q, r["S"], "o", ms=3, mfc=style["color"] if r["eps"] == 5.0 else "white",
                  mec=style["color"], mew=0.7, label=label_for(r), ls="none")
    for value, name in ((KAPPA, "\u03ba"), (G, "g")):
        ax_b.axvline(value, color="0.6", lw=0.5, ls=":")
        ax_b.text(value, 1.0, name, transform=ax_b.get_xaxis_transform(), ha="center", va="bottom", fontsize=7)
    ax_b.axhline(1.0, color="0.6", lw=0.5)
    ax_b.set_xlabel("|q| a$_M$")
    ax_b.set_ylabel("S(q) = \u27e8|\u03c1$_q$|\u00b2\u27e9 / N")
    ax_b.set_title("b   Static structure factor", pad=10)
    ax_b.legend(frameon=False, fontsize=6, loc="upper right")
    fig.savefig(path, dpi=300)
    plt.close(fig)


def stationarity_figure(runs, path, window=100):
    """Site occupations step by step: is the spatial pattern the same at the start and the end?

    The energy's blocking analysis says nothing about this: the walkers can
    mix quickly in energy and still move slowly between spatial patterns.
    """
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 3.6), sharex=True, sharey=True, constrained_layout=True)
    by_key = {(r["network"], r["eps"]): r for r in runs}
    shades = plt.get_cmap("Greys")(np.linspace(0.35, 0.95, observables.N_SITES))
    for row, network in enumerate(("CustomPsiformer", "SlaterNet")):
        for column, eps in enumerate((10.0, 5.0)):
            ax = axes[row, column]
            r = by_key.get((network, eps))
            if r is None:
                ax.set_axis_off()
                continue
            smooth = pd.DataFrame(r["occupation_series"]).rolling(window).mean().to_numpy()
            for k in range(observables.N_SITES):
                ax.plot(smooth[:, k], color=shades[k], lw=0.6)
            ax.axhline(r["N"] / observables.N_SITES, color=COLOURS[network], lw=0.8, ls="--")
            ax.set_title(label_for(r))
            if row == 1:
                ax.set_xlabel("evaluation step")
            if column == 0:
                ax.set_ylabel("site occupation")
    fig.suptitle(f"Occupation of each of the 9 sites during the evaluation (moving average over {window} steps;"
                 " dashed: 2/3, a uniform filling)", x=0.02, ha="left", fontsize=8)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def main():
    runs = [r for r in (load(name) for name in ORDER) if r is not None]
    if not runs:
        sys.exit(f"no evaluations found in {EVALUATION}")
    FIGURES.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    for table, name in ((energy_table(runs), "energies_meV_per_electron.csv"),
                        (structure_table(runs), "electron_structure.csv")):
        table.to_csv(RESULTS / name, index=False)
        with pd.option_context("display.width", 200, "display.precision", 4):
            print(f"\n{name}\n" + table.set_index("run").T.to_string())
    energy_figure(runs, FIGURES / "2_frozen_weight_energies.png")
    density_figure(runs, FIGURES / "3_electron_density.png")
    pair_correlation_figure(runs, FIGURES / "4_pair_correlation.png")
    order_figure(runs, FIGURES / "5_site_filling_and_structure_factor.png")
    stationarity_figure(runs, FIGURES / "6_site_occupation_during_evaluation.png")
    print(f"\ntables written to {RESULTS}, figures to {FIGURES}")


if __name__ == "__main__":
    main()
