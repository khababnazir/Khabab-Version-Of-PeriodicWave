"""
eps = 3, the authors' code from seed 42: its final weights FROZEN, next to the first eps = 3
self-attention run (clock seed): energy, density, pair correlation, S(kappa), sublattices, site
pattern.  Also S(kappa) and the sublattice shares of the walkers in EVERY checkpoint of the seed-42
run (when did the honeycomb form, and did it stay?).  (Until 2026-11-23 also the honeycomb-field
run; it is now in ../archive/eps3_seed42_honeycomb_field/, with the results it was in.)

Everything is measured with learn's functions, unchanged, through eps3_evaluation_results.py (the
same loading, tables and figure as for the first eps = 3 runs) and check_honeycomb.py (the walkers
of a checkpoint).  The training energy is the mean of the last 20 % of the iterations, 160,000 to
199,999, for both runs.

Run, from this folder (CPU only):
    $HOME/.local/bin/uv run --no-project --python ../../.venv/bin/python --with matplotlib --with pyblock python seed42_evaluation_results.py
Output, in ../runs/18_electrons_27_sites_eps3_seed42/:
    results/frozen_energies.csv, frozen_structure.csv, frozen_site_pattern.csv, checkpoint_structure.csv
    figures/frozen_density_pair_correlation.png
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SEED42 = HERE.parent / "runs" / "18_electrons_27_sites_eps3_seed42"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(SEED42))
import eps3_evaluation_results as ev3                 # noqa: E402  the first eps = 3 runs, unchanged
import check_honeycomb                                # noqa: E402  learn's walkers and measure

t1, e3, NAME = ev3.t1, ev3.e3, ev3.NAME
OUTPUT = Path("results") / "2deg-CoulombMoire" / "CustomPsiformer"
ORDINARY = SEED42 / "ordinary"
ev3.TITLES.update(seed42_ordinary="seed 42", authors_attention="first run (clock seed)")


def load_seed42(run, training, folder):
    """learn's load on <run>/evaluation/NAME, with the training energies of training/NAME."""
    t1.TRAINING, t1.EVALUATION, t1.GRADIENT = training, run / "evaluation", "real"
    loaded = t1.load(NAME)
    assert loaded is not None, f"{run / 'evaluation' / NAME}: no finished evaluation"
    assert loaded["final"] and loaded["step"] == 200_000 and loaded["eps"] == 3.0, folder
    assert loaded["network"] == "attention" and loaded["n_configurations"] == 20_480_000, folder
    return dict(loaded, folder=folder)


def checkpoint_structure():
    """S(kappa) and the sublattice shares of the 1024 walkers of every checkpoint of the seed-42 run."""
    cs, acr = check_honeycomb.cs, check_honeycomb.acr
    geometry = cs.geometry()
    cs.self_check(geometry)
    rows = []
    checkpoints = sorted((ORDINARY / OUTPUT / NAME).glob("qmcjax_ckpt_*.npz"))
    assert checkpoints, ORDINARY
    for checkpoint in checkpoints:
        walkers = acr.walkers(checkpoint, geometry[0])
        rows.append(dict(run="seed42", iteration=int(checkpoint.stem.split("_")[-1]), walkers=len(walkers),
                         **cs.measure(walkers, *geometry)))
    return pd.DataFrame(rows)


def main():
    first = ev3.load_authors("CustomPsiformer")
    seed42 = load_seed42(ORDINARY, ORDINARY / OUTPUT, "seed42_ordinary")
    runs = [seed42, first]
    assert all(len(r["training_series"]) == 40_000 for r in runs), "last 20 % = 40,000 iterations each"
    (SEED42 / "results").mkdir(exist_ok=True)
    (SEED42 / "figures").mkdir(exist_ok=True)

    energies = e3.energy_table(runs)
    energies["frozen_minus_training"] = energies["E_frozen"] - energies["E_training_last20pct"]
    energies["frozen_minus_first_run"] = energies["E_frozen"] - first["frozen"]
    energies["frozen_minus_first_run_error"] = np.hypot(energies["E_frozen_error"], first["frozen_error"])
    structure, pattern = t1.structure_table(runs), e3.site_pattern_table(runs)
    trajectory = checkpoint_structure()
    energies.to_csv(SEED42 / "results" / "frozen_energies.csv", index=False)
    structure.to_csv(SEED42 / "results" / "frozen_structure.csv", index=False)
    pattern.to_csv(SEED42 / "results" / "frozen_site_pattern.csv", index=False)
    trajectory.to_csv(SEED42 / "results" / "checkpoint_structure.csv", index=False)
    ev3.density_pair_figure(runs, SEED42 / "figures" / "frozen_density_pair_correlation.png",
                            title=ev3.TITLE.replace("supercell). ", "supercell).\n"))   # two columns: two lines

    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(energies[["run", "E_frozen", "E_frozen_error", "E_training_last20pct", "E_training_last20pct_error",
                        "frozen_minus_training", "frozen_minus_first_run", "frozen_minus_first_run_error",
                        "kinetic", "moire", "coulomb", "halves_apart_in_errors", "settled", "pmove"]].to_string(index=False))
        print(structure.drop(columns=["electrons", "sites", "eps"]).to_string(index=False))
        print(trajectory.groupby("run", sort=False)[["S_kappa", "sublattice_0", "sublattice_1", "sublattice_2"]]
              .agg(["min", "max"]).round(3).to_string())
    for name, group in pattern.groupby("run", sort=False):
        print(f"{name}: site pattern summed over its 27 wave vectors {group['site_pattern'].sum():.4f}, "
              f"at +-kappa {group.loc[group['at_kappa'], 'site_pattern'].sum():.4f}")


if __name__ == "__main__":
    main()
