"""
eps = 3, the authors' code unchanged: training energy and the structure of the walkers at every
checkpoint, for the self-attention network AND SlaterNet.

learn/stage08_own_code/analysis/18_electrons_27_sites/authors_code_results.py does this for the
eps = 5 and 10 runs, but cannot read these two: it takes the network sizes from the
CustomPsiformer entry of config.json (SlaterNet has none) and looks up the paper's Table I value,
which exists only for eps = 5 and 10.  This script calls ITS functions, unchanged -- the training
energy over the last 20 % of the iterations with a blocking error, the walkers of each checkpoint
(with its layout guard), and checkpoint_structure.measure (S(kappa), sublattice shares, ...) --
and only leaves out those two things.

Run, from this folder (CPU only):
    $HOME/.local/bin/uv run --no-project --python ../../.venv/bin/python --with matplotlib --with pyblock python eps3_checkpoint_structure.py
Output, in ../runs/18_electrons_27_sites_eps3_released_settings/analysis/:
    summary.csv, checkpoint_structure_CustomPsiformer.csv, checkpoint_structure_SlaterNet.csv
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "learn" / "stage08_own_code" / "analysis" / "18_electrons_27_sites"))
import authors_code_results as acr                    # noqa: E402  learn's, unchanged
cs = acr.cs

RUN_FOLDER = HERE.parent / "runs" / "18_electrons_27_sites_eps3_released_settings"
EPS = 3.0


def analyse(run, geo):
    config = json.loads((run / "config.json").read_text())
    network = config["network"]["network_type"]
    scale = config["system"]["make_local_energy_kwargs"]["potential_kwargs"]["interaction_energy_scale"]
    eps = round(5.0 * acr.U_AT_EPS_5 / scale, 6)
    assert eps == EPS, (run, eps)
    row = dict(network=network, eps=eps, walkers=config["batch_size"], iterations=config["optim"]["iterations"],
               learning_rate=config["optim"]["lr"]["rate"], determinants=config["network"]["determinants"])
    row.update(acr.training_energy(run, row["iterations"]))
    rows, pooled = [], []
    for iteration, R in acr.checkpoint_rows(run, geo):
        rows.append(dict(iteration=iteration, walkers=len(R), **cs.measure(R, *geo)))
        if iteration >= acr.final_window_start(row["iterations"]):
            pooled.append(R)
    row.update(checkpoints=len(rows), pooled_checkpoints=len(pooled), pooled_walkers=sum(len(R) for R in pooled))
    measured = cs.measure(np.concatenate(pooled), *geo)
    row.update({"pooled_" + name: measured[name] for name in acr.STRUCTURE})
    return row, rows


def write_csv(path, rows):
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    geo = cs.geometry()
    cs.self_check(geo)
    print("self-check passed: honeycombs 4.5, even filling 0")
    runs = acr.their_runs(RUN_FOLDER)
    assert len(runs) == 2, runs
    out = RUN_FOLDER / "analysis"
    out.mkdir(exist_ok=True)
    summary = []
    for run in runs:
        row, rows = analyse(run, geo)
        summary.append(row)
        write_csv(out / f"checkpoint_structure_{row['network']}.csv", rows)
        S = [r["S_kappa"] for r in rows]
        print(f"{row['network']}: finished {row['finished']}, last step {row['last_step']}, non-finite rows {row['non_finite']}")
        print(f"   energy, last 20 % ({row['rows_in_window']} rows): {row['energy']:.4f} +- {row['error']:.4f} meV per electron")
        print(f"   S(kappa) over {len(rows)} checkpoints: {min(S):.3f} to {max(S):.3f}; last 20 % pooled "
              f"({row['pooled_walkers']} walkers): S(kappa) {row['pooled_S_kappa']:.3f}, sublattices "
              f"{row['pooled_sublattice_0']:.3f} {row['pooled_sublattice_1']:.3f} {row['pooled_sublattice_2']:.3f}, "
              f"emptiest <= 3: {row['pooled_emptiest_at_most_3']:.3f}")
        print("   last 5 checkpoints: " + "  ".join(f"{r['iteration']}: S {r['S_kappa']:.2f}, sub "
              f"{r['sublattice_0']:.2f}/{r['sublattice_1']:.2f}/{r['sublattice_2']:.2f}" for r in rows[-5:]))
    write_csv(out / "summary.csv", summary)


if __name__ == "__main__":
    main()
