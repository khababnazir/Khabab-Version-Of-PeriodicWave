"""
Stage 8 -- figures and table from the frozen-weight evaluations.

Run after evaluate_weights.py (CPU is enough; matplotlib is layered on top of
the project environment without changing it):
    uv run --no-project --python ../../.venv/bin/python --with matplotlib python plot_weights.py

Reads runs/stage08/evaluation/<run>/ and writes into runs/stage08/evaluation/:
    evaluation_summary.csv          one row per run, all numbers below
    evaluation_energy.png           (a) frozen vs training vs Fig. 5, (b) blocking curves
    evaluation_density.png          n(r) for each run                   -- paper Fig. 6(a, b)
    evaluation_pair_correlation.png g(dr) maps and g(r)                 -- paper Fig. 6(c, d)
    evaluation_order.png            (a) which sites are filled, (b) structure factor S(q)

Comparing with Fig. 6.  The paper's Fig. 6 is for 27 sites (18 electrons);
our runs are the 9-site, 6-electron system of Fig. 5.  The comparison is
qualitative: does eps = 10 look like a liquid and eps = 5 like a crystal?
"""
import json
import sys
from pathlib import Path

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

RUNS = HERE.parents[1] / "runs" / "stage08"
EVALUATION = RUNS / "evaluation"
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
    n_configurations = summary["n_configurations"]

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
    training = describe(RUNS / name)
    run["training"], run["training_error"] = training["mean"], training["error"]
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

    # Site-pair correlation: <n_k n_l> / <n>^2 averaged over pairs at one distance.
    # Honeycomb crystal: 0.75 for neighbours, 1.5 at sqrt(3).  Random filling: 15/16 for both.
    sites = observables.moire_sites(supercell)
    distance = np.array([[float(np.linalg.norm(observables.minimum_image(np.asarray(a - b), supercell)))
                          for b in sites] for a in sites])
    products = totals["occupation_products"] / n_configurations
    mean_squared = run["occupation"].mean() ** 2
    run["g_site_neighbours"] = products[np.isclose(distance, 1.0)].mean() / mean_squared
    run["g_site_sqrt3"] = products[np.isclose(distance, np.sqrt(3))].mean() / mean_squared

    # Structure factor at the supercell's wave vectors.
    run["q"] = np.array(summary["q_vectors"])
    run["S"], run["S_connected"] = observables.structure_factor(
        totals["rho_q"], totals["rho_q_squared"], n_configurations, N)
    at_kappa = np.isclose(np.linalg.norm(run["q"], axis=1), KAPPA, atol=1e-6)
    assert at_kappa.sum() == 6, "expected the 6 corners of the moire Brillouin zone"
    run["S_kappa"], run["S_kappa_connected"] = run["S"][at_kappa].mean(), run["S_connected"][at_kappa].mean()
    return run


def summary_table(runs):
    rows = []
    for r in runs:
        rows.append(dict(
            run=r["name"], network=LABELS[r["network"]], eps=r["eps"],
            evaluation_steps=r["summary"]["steps_done"], configurations=r["n_configurations"],
            E_frozen=r["frozen"], E_frozen_error=r["frozen_error"],
            E_training_last20=r["training"], E_training_error=r["training_error"],
            E_fig5_by_eye=r["fig5"], kinetic=r["kinetic"], moire=r["moire"], coulomb=r["coulomb"],
            sigma_local_energy_meV=r["sigma_local_energy_meV"], pmove=r["pmove"],
            P_honeycomb=r["p_honeycombs"].sum(), P_largest_honeycomb=r["p_honeycombs"].max(),
            P_honeycomb_if_random=r["p_honeycombs_if_random"],
            P_double_occupancy=r["p_double"],
            g_site_neighbours=r["g_site_neighbours"], g_site_sqrt3=r["g_site_sqrt3"],
            occupation_min=r["occupation"].min(), occupation_max=r["occupation"].max(),
            S_kappa=r["S_kappa"], S_kappa_connected=r["S_kappa_connected"],
            steps_per_second=r["summary"]["steps_per_second"]))
    return pd.DataFrame(rows)


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
    ax_a.set_title("a   Frozen weights reproduce the training energies,\n     within the Fig. 5 reading accuracy")
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
                 label="pair correlation g(\u0394r)  (1 = uncorrelated)")

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
    ax_a.set_title("a   Honeycomb fillings are no more common than at random")
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
    ax_b.set_title("b   No crystal peak at \u03ba in any run", pad=10)
    ax_b.legend(frameon=False, fontsize=6, loc="upper right")
    fig.savefig(path, dpi=300)
    plt.close(fig)


def main():
    runs = [r for r in (load(name) for name in ORDER) if r is not None]
    if not runs:
        sys.exit(f"no evaluations found in {EVALUATION}")
    table = summary_table(runs)
    table.to_csv(EVALUATION / "evaluation_summary.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 30, "display.precision", 4):
        print(table.T.to_string(header=False))
    energy_figure(runs, EVALUATION / "evaluation_energy.png")
    density_figure(runs, EVALUATION / "evaluation_density.png")
    pair_correlation_figure(runs, EVALUATION / "evaluation_pair_correlation.png")
    order_figure(runs, EVALUATION / "evaluation_order.png")
    print(f"figures written to {EVALUATION}")


if __name__ == "__main__":
    main()
