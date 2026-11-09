"""
Stage 8 -- does a crystal form DURING training?  S(kappa) at every checkpoint (18 electrons, 27 sites).

The frozen-weight evaluation (evaluate_weights.py) measures a trained network once, at the end.
Every training checkpoint also holds the 4,096 walkers of that iteration: one sample of |Psi|^2
each.  This script measures them at every checkpoint:

  S_kappa             <|rho_kappa|^2> / N with rho_kappa = sum_j exp(-i kappa . r_j), averaged
                      over the walkers and the 6 corners kappa of the moire Brillouin zone.
                      Squared BEFORE averaging, so it sees a honeycomb even when different
                      walkers leave different sublattices empty.
  S_kappa_connected   the same minus |<rho_kappa>|^2 / N: without the part the density shows.
  S_sites             S_kappa with every electron moved onto its nearest moire site.  It then
                      depends only on the numbers A, B, C of electrons on the three sublattices:
                      ((A - B)^2 + (B - C)^2 + (C - A)^2) / (2N).
  sublattice_0..2     the share of the electrons on each sublattice (1/3 each: no pattern in the
                      density; 0, 1/2, 1/2: a honeycomb that stays put).  The sublattice of a
                      site n1 a1 + n2 a2 is (n1 + n2) mod 3, as in table1_results.py.
  emptiest_at_most_3  the share of walkers whose emptiest sublattice holds 3 electrons or fewer.

Reference values for electrons exactly on sites (18 electrons, 27 sites): a honeycomb (0, 9, 9)
gives 4.5; 18 of the 27 sites filled at random, one electron each, 1 - 17/26 = 0.35;
independent electrons 1.  Electrons spread around their sites pull every value towards 1.

Run, from this folder, on the CPU (it reads only complete checkpoints -- train_moire.py writes
each one under a temporary name and then renames it -- so it can run while a run trains):
    JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= uv run --no-project --python ../../../../.venv/bin/python \\
        --with matplotlib --with pyblock python checkpoint_structure.py RUN [RUN ...]
RUN is a training folder, e.g.
    ../../../../runs/stage08/18_electrons_27_sites_eps3_real_gradient/training/attention_18e_eps3
Output: results/checkpoint_structure_<run>.csv in each run's system folder, and
figures/checkpoint_structure.png in the system folder of the FIRST run, all runs in one figure.
Every start first checks the measurement on ideal honeycombs and on an even filling.
"""
import argparse
import csv
import json
import os
import sys
import zipfile
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax                                            # noqa: E402
import matplotlib                                     # noqa: E402
import numpy as np                                    # noqa: E402

jax.config.update("jax_enable_x64", True)
matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))                  # observables.py
sys.path.insert(0, str(HERE.parents[1]))              # train_moire.py, which adds the stage folders
import train_moire                                    # noqa: E402
import observables                                    # noqa: E402
import slaternet                                      # noqa: E402
from table1_results import sublattice_of_sites        # noqa: E402  (this folder)

N = 18
KAPPA = 4 * np.pi / 3                                 # corner of the moire Brillouin zone (a_M = 1)
FIELDS = ["iteration", "S_kappa", "S_kappa_connected", "S_sites",
          "sublattice_0", "sublattice_1", "sublattice_2", "emptiest_at_most_3"]


def geometry():
    """Supercell, the 27 sites, the 6 kappas and the sublattice of every site."""
    supercell = np.asarray(slaternet.supercell_vectors(train_moire.SITES[N]), dtype=np.float64)
    sites = np.asarray(observables.moire_sites(supercell))
    qs, _ = observables.supercell_q_vectors(supercell, q_max=KAPPA + 0.1)
    qs = np.asarray(qs)
    kappas = qs[np.isclose(np.linalg.norm(qs, axis=1), KAPPA, atol=1e-6)]
    sublattice = sublattice_of_sites(supercell)
    assert len(sites) == 27 and len(kappas) == 6, (len(sites), len(kappas))
    assert np.bincount(sublattice).tolist() == [9, 9, 9]
    return supercell, sites, kappas, sublattice


def measure(R, supercell, sites, kappas, sublattice):
    """The numbers above for one checkpoint's walkers R, shape (walkers, N, 2)."""
    rho = np.asarray(observables.density_fourier(R, kappas))                     # (walkers, 6)
    S = np.mean(np.abs(rho) ** 2) / N
    S_connected = S - np.mean(np.abs(rho.mean(axis=0)) ** 2) / N
    occupations = np.asarray(observables.site_occupations(R, supercell))         # (walkers, 27)
    counts = np.stack([occupations[:, sublattice == g].sum(axis=1) for g in range(3)], axis=1)
    a, b, c = counts.T
    S_sites = np.mean(((a - b) ** 2 + (b - c) ** 2 + (c - a) ** 2) / 2) / N
    rho_sites = occupations @ np.exp(-1j * sites @ kappas.T)                     # electrons on their sites
    assert np.isclose(np.mean(np.abs(rho_sites) ** 2) / N, S_sites), "S_sites: the two formulas disagree"
    share = counts.mean(axis=0) / N
    return dict(S_kappa=S, S_kappa_connected=S_connected, S_sites=S_sites,
                sublattice_0=share[0], sublattice_1=share[1], sublattice_2=share[2],
                emptiest_at_most_3=np.mean(counts.min(axis=1) <= 3))


def self_check(geo):
    """Ideal arrangements, worked out by hand.  Stops the script if any number is off."""
    _, sites, _, sublattice = geo
    empty_0 = sites[sublattice != 0]                  # honeycomb: sublattice 0 empty, 1 and 2 full
    empty_1 = sites[sublattice != 1]                  # the honeycomb that leaves sublattice 1 empty
    m = measure(np.stack([empty_0, empty_1, empty_0, empty_1]), *geo)
    # Every walker is a honeycomb: S = 4.5.  Half leave 0 empty, half 1: rho_kappa is -9 p_0 or
    # -9 p_1 (p_g = the phase of sublattice g, p_0 + p_1 + p_2 = 0), its mean is 4.5 p_2, so the
    # density holds |4.5|^2 / 18 = 1.125 of the 4.5 and the connected part is 3.375.
    expected = dict(S_kappa=4.5, S_kappa_connected=3.375, S_sites=4.5,
                    sublattice_0=0.25, sublattice_1=0.25, sublattice_2=0.5, emptiest_at_most_3=1.0)
    for name, value in expected.items():
        assert np.isclose(m[name], value, atol=1e-9), f"self-check, honeycombs: {name} = {m[name]}, not {value}"
    even = np.concatenate([sites[sublattice == g][:6] for g in range(3)])        # 6, 6, 6
    m = measure(even[None], *geo)
    for name, value in dict(S_kappa=0.0, S_sites=0.0, emptiest_at_most_3=0.0).items():
        assert np.isclose(m[name], value, atol=1e-9), f"self-check, even filling: {name} = {m[name]}, not {value}"


def checkpoints(run):
    """(iteration, walkers) of every readable checkpoint, oldest first."""
    for path in sorted(run.glob("checkpoint_*.npz")):
        try:
            with np.load(path) as saved:
                R = saved["walkers"].astype(np.float64)
        except (OSError, ValueError, KeyError, zipfile.BadZipFile) as error:
            print(f"   {path.name}: unreadable ({error}); skipped")
            continue
        assert R.ndim == 3 and R.shape[1:] == (N, 2), f"{path}: walkers of shape {R.shape}"
        yield int(path.stem.split("_")[1]), R


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("runs", nargs="+", type=Path, help="training folders")
    runs = [run.resolve() for run in parser.parse_args().runs]
    geo = geometry()
    self_check(geo)
    print("self-check passed: honeycombs 4.5, even filling 0")

    fig, (ax_s, ax_share) = plt.subplots(1, 2, figsize=(10, 4.4), layout="constrained")
    for run in runs:
        settings = json.loads((run / "settings.json").read_text())
        assert settings["electrons"] == N, f"{run}: not an 18-electron run"
        rows = [dict(iteration=done, **measure(R, *geo)) for done, R in checkpoints(run)]
        assert rows, f"{run}: no readable checkpoint"
        system = run.parent.parent                      # <system>/training/<run>
        out = system / "results" / f"checkpoint_structure_{run.name}.csv"
        out.parent.mkdir(exist_ok=True)
        with open(out, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows({k: (v if k == "iteration" else f"{v:.6f}") for k, v in row.items()} for row in rows)
        last = rows[-1]
        print(f"{run.name} ({settings['gradient']} gradient, {system.name}): {len(rows)} checkpoints; "
              f"last, iteration {last['iteration']}: S_kappa {last['S_kappa']:.3f}, S_sites {last['S_sites']:.3f}, "
              f"sublattices {last['sublattice_0']:.3f} / {last['sublattice_1']:.3f} / {last['sublattice_2']:.3f}"
              f"  -> {out}")
        it = np.array([r["iteration"] for r in rows]) / 1000
        form = settings["settings"].get("network_form", "released")     # missing before 2026-11-08: released
        label = f"{settings['network']} ({form} network), ε = {settings['eps']:g}, {settings['gradient']} gradient"
        line, = ax_s.plot(it, [r["S_kappa"] for r in rows], marker=".", label=label)
        ax_share.plot(it, [min(r["sublattice_0"], r["sublattice_1"], r["sublattice_2"]) for r in rows],
                      marker=".", color=line.get_color())
    # The reference lines go into the legend below the panels, so nothing is drawn over the data.
    lines = list(ax_s.get_lines())
    highest = max(max(line.get_ydata()) for line in lines)
    lines.append(ax_s.axhline(1.0, color="0.45", ls="--", lw=0.9,
                              label="a: independent electrons, S(κ) = 1  (a honeycomb, electrons on sites: 4.5)"))
    lines.append(ax_share.axhline(1 / 3, color="0.45", ls=":", lw=1.2,
                                  label="b: 1/3, no pattern in the density  (0: a honeycomb that stays put)"))
    ax_s.set_ylim(top=max(1.05, 1.05 * highest))
    ax_s.set(xlabel="iteration (thousands)", ylabel="S(κ)", title="a   S(κ) at every checkpoint")
    ax_share.set_ylim(0, 0.36)
    ax_share.set(xlabel="iteration (thousands)", ylabel="share of the electrons",
                 title="b   share on the emptiest sublattice")
    fig.legend(handles=lines, loc="outside lower center", ncol=2, fontsize=8, frameon=False)
    figure = runs[0].parent.parent / "figures" / "checkpoint_structure.png"
    figure.parent.mkdir(exist_ok=True)
    fig.savefig(figure, dpi=150)
    print(f"figure: {figure}")


if __name__ == "__main__":
    main()
