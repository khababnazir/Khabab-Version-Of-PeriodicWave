"""
Stage 8 -- what did the AUTHORS' OWN code produce?  (18 electrons, 27 sites)

The reference runs in runs/stage08/18_electrons_27_sites_authors_code*/ were trained by the
authors' released code alone.  This script only READS what their code wrote, in
<reference folder>/results/2deg-CoulombMoire/CustomPsiformer/<parameters>/:

  train_stats.csv          one row per iteration; `energy` is the mean local energy of that
                           iteration's walkers, in the code's unit for all 18 electrons.
  qmcjax_ckpt_<step>.npz   a checkpoint every 30 minutes; data["positions"] holds the walkers of
                           that step, one row per walker: x1, y1, x2, y2, ... (their networks.py
                           reshapes it to (electrons, 2)).
  config.json              the settings the run was started with.

It reports, for every run found:

  energy      the mean of `energy` over the last 20 % of the planned iterations, in meV per
              electron (x 3.375559 / 18), with a blocking error bar, next to the paper's Table I.
              This is a TRAINING average: the weights still change a little from row to row.  It
              is the same kind of number as "training mean" in table1_results.py, not a
              frozen-weight evaluation.
  structure   S(kappa), the sublattice shares and the other numbers of checkpoint_structure.py
              -- measured by the SAME function as for our own runs -- at every checkpoint, and
              once more for the walkers of all checkpoints in the last 20 % pooled together.
              A honeycomb gives S(kappa) = 4.5 and sublattice shares 0, 1/2, 1/2.

Run, from this folder, on the CPU (it can run while a run trains: an unfinished run is marked):
    JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= uv run --no-project --python ../../../../.venv/bin/python \\
        --with matplotlib --with pyblock python authors_code_results.py REFERENCE [REFERENCE ...]
REFERENCE is e.g. ../../../../runs/stage08/18_electrons_27_sites_authors_code
Output, in <REFERENCE>/analysis/: summary.csv (one row per run), checkpoint_structure_eps<eps>.csv
and checkpoint_structure.png.  Tests: test_authors_code_results.py.
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import matplotlib                                     # noqa: E402
import numpy as np                                    # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))                  # observables.py
sys.path.insert(0, str(HERE.parent / "6_electrons_9_sites"))
import observables                                    # noqa: E402
from analyse_runs import ENERGY_UNIT_MEV, blocked_mean   # noqa: E402
import checkpoint_structure as cs                     # noqa: E402  (this folder)

N = 18
FINAL_PART = 5                                        # the last 1/5 = 20 % of the iterations
U_AT_EPS_5 = 10.62347336201578                        # their interaction_energy_scale at eps = 5
PAPER = {10.0: (-32.070, 0.007), 5.0: (-59.127, 0.009)}    # Table I, self-attention network
STRUCTURE = ["S_kappa", "S_kappa_connected", "S_sites", "sublattice_0", "sublattice_1",
             "sublattice_2", "emptiest_at_most_3"]


def their_runs(reference):
    """The folders their script wrote below one reference folder, one per set of parameters."""
    return sorted(path.parent for path in (reference / "results").glob("*/*/*/config.json"))


def settings(run):
    """The settings that tell the runs apart, from their config.json."""
    config = json.loads((run / "config.json").read_text())
    scale = config["system"]["make_local_energy_kwargs"]["potential_kwargs"]["interaction_energy_scale"]
    network = config["network"]["CustomPsiformer"]
    return dict(eps=round(5.0 * U_AT_EPS_5 / scale, 6), walkers=config["batch_size"],
                iterations=config["optim"]["iterations"], learning_rate=config["optim"]["lr"]["rate"],
                layers=network["num_layers"], heads=network["num_heads"])


def final_window_start(iterations):
    """The first iteration of the last 20 %."""
    return iterations - iterations // FINAL_PART


def training_energy(run, iterations):
    """Mean energy per electron (meV) over the last 20 % of the planned iterations."""
    text = (run / "train_stats.csv").read_text()
    lines = text.split("\n")
    if not text.endswith("\n"):                       # the run is writing its last row right now
        lines = lines[:-1]
    lines = [line for line in lines if line]
    if len(lines) < 2:
        return dict(last_step=-1, finished=False, rows_in_window=0, energy=np.nan, error=np.nan, non_finite=0)
    stats = np.genfromtxt(lines, delimiter=",", names=True)
    step = np.atleast_1d(stats["step"]).astype(int)
    assert np.array_equal(step, np.arange(len(step))), f"{run}: the steps are not 0, 1, 2, ... (restarted twice?)"
    energy = np.atleast_1d(stats["energy"]) * ENERGY_UNIT_MEV / N
    window = energy[step >= final_window_start(iterations)]
    mean, error = blocked_mean(window[np.isfinite(window)])
    return dict(last_step=int(step[-1]), finished=bool(step[-1] == iterations - 1),
                rows_in_window=len(window), energy=mean, error=error,
                non_finite=int(np.sum(~np.isfinite(energy))))


def nearest_distance(R, supercell):
    """The mean distance from an electron to its nearest other electron (shortest periodic image)."""
    d = np.asarray(observables.minimum_image(R[:, :, None, :] - R[:, None, :, :], supercell))
    r = np.linalg.norm(d, axis=-1) + np.where(np.eye(R.shape[1], dtype=bool), np.inf, 0.0)
    return float(r.min(axis=2).mean())


def walkers(path, supercell):
    """The walkers of one of their checkpoints, shape (walkers, 18, 2), in units of a_M."""
    with np.load(path, allow_pickle=True) as saved:
        raw = np.asarray(saved["data"].item()["positions"], dtype=np.float64)
    assert raw.shape[-1] == 2 * N, f"{path}: positions of shape {raw.shape}"
    R = raw.reshape(-1, N, 2)                          # x1, y1, x2, y2, ...
    # Guard against reading the file the wrong way round (x1 .. x18, y1 .. y18): that would turn
    # every picture into noise and hide a crystal.  Electrons keep apart; scrambled points do not.
    other = raw.reshape(-1, 2, N).transpose(0, 2, 1)
    here, there = nearest_distance(R, supercell), nearest_distance(other, supercell)
    assert here > 1.2 * there, f"{path}: electrons are no further apart ({here:.3f}) than scrambled points ({there:.3f})"
    return R


def checkpoint_rows(run, geo):
    """(iteration, walkers) of every checkpoint of one run, oldest first."""
    for path in sorted(run.glob("qmcjax_ckpt_*.npz")):
        yield int(path.stem.split("_")[-1]), walkers(path, geo[0])


def analyse(run, geo):
    """Everything this script reports for one run: a summary row and the per-checkpoint rows."""
    row = settings(run)
    row.update(training_energy(run, row["iterations"]))
    paper, paper_error = PAPER[row["eps"]]
    row.update(paper=paper, paper_error=paper_error, minus_paper=row["energy"] - paper)
    rows, pooled = [], []
    for iteration, R in checkpoint_rows(run, geo):
        rows.append(dict(iteration=iteration, walkers=len(R), **cs.measure(R, *geo)))
        if iteration >= final_window_start(row["iterations"]):
            pooled.append(R)
    row.update(checkpoints=len(rows), pooled_checkpoints=len(pooled),
               pooled_walkers=sum(len(R) for R in pooled))
    measured = cs.measure(np.concatenate(pooled), *geo) if pooled else {name: np.nan for name in STRUCTURE}
    row.update({"pooled_" + name: measured[name] for name in STRUCTURE})
    return row, rows


def write_csv(path, rows):
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def figure(results, path):
    """S(kappa) and the sublattice shares at every checkpoint, one colour per run."""
    fig, (ax_s, ax_share) = plt.subplots(1, 2, figsize=(10, 4.4), layout="constrained")
    for colour, (row, rows) in zip(["tab:blue", "tab:orange", "tab:green"], results):
        if not rows:
            continue
        x = [r["iteration"] / 1000 for r in rows]
        label = f"eps = {row['eps']:g}, {row['walkers']} walkers"
        ax_s.plot(x, [r["S_kappa"] for r in rows], "o-", ms=3, color=colour, label=label)
        for g, style in enumerate(["-", "--", ":"]):
            ax_share.plot(x, [r[f"sublattice_{g}"] for r in rows], style, color=colour,
                          label=label if g == 0 else None)
    ax_s.axhline(1.0, color="grey", lw=0.8, ls=":")
    ax_s.set(xlabel="iteration (thousands)", ylabel="S(kappa)", ylim=(0, 1.5),
             title="a honeycomb would give 4.5 (off this scale)")
    ax_share.axhline(1 / 3, color="grey", lw=0.8, ls=":")
    ax_share.set(xlabel="iteration (thousands)", ylabel="share of the electrons on each sublattice",
                 ylim=(0, 0.55), title="a honeycomb would give 0, 1/2, 1/2")
    ax_s.legend(frameon=False)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("references", nargs="+", type=Path, help="reference folders")
    geo = cs.geometry()
    cs.self_check(geo)
    print("self-check passed: honeycombs 4.5, even filling 0")
    for reference in parser.parse_args().references:
        reference = reference.resolve()
        results = [analyse(run, geo) for run in their_runs(reference)]
        out = reference / "analysis"
        out.mkdir(exist_ok=True)
        write_csv(out / "summary.csv", [row for row, _ in results])
        for row, rows in results:
            if rows:
                write_csv(out / f"checkpoint_structure_eps{row['eps']:g}.csv", rows)
        figure(results, out / "checkpoint_structure.png")
        print(f"\n{reference.name}")
        for row, rows in results:
            state = "finished" if row["finished"] else f"NOT finished (step {row['last_step']} of {row['iterations']})"
            print(f"  eps = {row['eps']:g} | {row['walkers']} walkers, {row['layers']} layers, {row['heads']} heads,"
                  f" learning rate {row['learning_rate']:g} | {state}")
            print(f"    energy, last 20 % ({row['rows_in_window']} rows): {row['energy']:.4f} +- {row['error']:.4f} meV"
                  f" | paper {row['paper']:.3f}({row['paper_error'] * 1000:.0f}) | this run - paper {row['minus_paper']:+.4f}"
                  f" | non-finite rows {row['non_finite']}")
            if rows:
                values = [r["S_kappa"] for r in rows]
                print(f"    S(kappa) over {len(rows)} checkpoints: {min(values):.3f} to {max(values):.3f}"
                      f" | last 20 % pooled ({row['pooled_walkers']} walkers): S(kappa) {row['pooled_S_kappa']:.3f},"
                      f" sublattices {row['pooled_sublattice_0']:.3f} {row['pooled_sublattice_1']:.3f}"
                      f" {row['pooled_sublattice_2']:.3f}, emptiest <= 3: {row['pooled_emptiest_at_most_3']:.3f}")
        print(f"  written: {out}")


if __name__ == "__main__":
    main()
