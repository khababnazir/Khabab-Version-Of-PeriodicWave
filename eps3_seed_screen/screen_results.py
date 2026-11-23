"""
eps = 3 seed screen: did each short start end in the honeycomb?

For every FINISHED start of both codes (5,000 iterations), the walkers of its final checkpoint ->
S(kappa), the three sublattice shares and the rest of learn's checkpoint_structure.measure
(unchanged; the authors' checkpoints read with authors_code_results.walkers, with its layout guard,
ours with checkpoint_structure.checkpoints), and the training energy averaged over iterations 4,000
to 4,999, in meV per electron.  A start counts as a honeycomb when its emptiest sublattice holds
less than 15 % of the electrons: a honeycomb holds about 3 % there, the even, liquid-like state
30 % or more.  As a control, the same numbers for two checkpoints whose answer is known: the
authors' seed-42 run at 4,709 (a honeycomb) and their first eps = 3 run at 4,759 (not).

Run, from this folder (CPU only; it reads only finished starts, so it can run during the screen):
    JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= $HOME/.local/bin/uv run --no-project --python ../.venv/bin/python \
        --with matplotlib --with pyblock python screen_results.py
Output: results/screen.csv (one row per start, the two controls first).
"""
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "learn" / "stage08_own_code" / "analysis" / "18_electrons_27_sites"))
import authors_code_results as acr                    # noqa: E402  learn's, unchanged

cs = acr.cs
UNIT = 3.375559 / 18                                  # code energy (whole system) -> meV per electron
OUT = Path("results/2deg-CoulombMoire/CustomPsiformer/el18_0_N27_V4.44370908_45.0_U17.70578894")
ITERATIONS = 5000
HONEYCOMB_BELOW = 0.15
RUNS = REPO / "authors_exact_code_runs" / "runs"
CONTROLS = [("authors, seed 42 run (control)", RUNS / "18_electrons_27_sites_eps3_seed42/ordinary" / OUT, 4709),
            ("authors, first run (control)", RUNS / "18_electrons_27_sites_eps3_released_settings" / OUT, 4759)]


def seeds(code):
    return sorted(int(p.name.split("_")[1]) for p in (HERE / code).glob("seed_*"))


def training(train_stats):
    """Mean energy (meV per electron) and pmove over iterations 4,000 to 4,999; None if unfinished."""
    try:
        d = pd.read_csv(train_stats, usecols=["step", "energy", "pmove"])
    except pd.errors.EmptyDataError:              # a start that has only just begun
        return None
    if len(d) != ITERATIONS or list(d["step"]) != list(range(ITERATIONS)):
        return None
    last = d[d["step"] >= ITERATIONS - 1000]
    return last["energy"].mean() * UNIT, last["pmove"].mean()


def row(code, seed, iteration, walkers, energy, pmove, geometry):
    m = cs.measure(walkers, *geometry)
    shares = [m["sublattice_0"], m["sublattice_1"], m["sublattice_2"]]
    return dict(code=code, seed=seed, iteration=iteration, walkers=len(walkers), **m,
                emptiest_sublattice=min(shares), honeycomb=min(shares) < HONEYCOMB_BELOW,
                E_4000_4999_meV=energy, pmove_4000_4999=pmove)


def main():
    geometry = cs.geometry()
    cs.self_check(geometry)
    rows = []
    for name, folder, iteration in CONTROLS:
        rows.append(row(name, None, iteration, acr.walkers(folder / f"qmcjax_ckpt_{iteration:06d}.npz", geometry[0]),
                        None, None, geometry))
    for seed in seeds("authors"):
        folder = HERE / "authors" / f"seed_{seed}" / OUT
        final = folder / f"qmcjax_ckpt_{ITERATIONS - 1:06d}.npz"
        done = training(folder / "train_stats.csv") if (folder / "train_stats.csv").exists() else None
        if final.exists() and done:
            rows.append(row("authors", seed, ITERATIONS, acr.walkers(final, geometry[0]), *done, geometry))
    for seed in seeds("ours"):
        folder = HERE / "ours" / f"seed_{seed}"
        done = training(folder / "train_stats.csv") if (folder / "train_stats.csv").exists() else None
        final = [R for iteration, R in cs.checkpoints(folder) if iteration == ITERATIONS]
        if final and done:
            rows.append(row("ours", seed, ITERATIONS, final[0], *done, geometry))
    table = pd.DataFrame(rows)
    (HERE / "results").mkdir(exist_ok=True)
    table.to_csv(HERE / "results" / "screen.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_rows", 100):
        print(table[["code", "seed", "S_kappa", "sublattice_0", "sublattice_1", "sublattice_2",
                     "honeycomb", "E_4000_4999_meV"]].round(4).to_string(index=False))
    for code in ("authors", "ours"):
        part = table[table["code"] == code]
        print(f"{code}: {len(part)} starts finished, {int(part['honeycomb'].sum())} honeycomb")


if __name__ == "__main__":
    main()
