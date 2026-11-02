"""
Stage 8: start a new training folder from ONE checkpoint of an existing run.

Made for the gradient test of 2026-11-02 (runs/stage08/18_electrons_27_sites/
gradient_test/): SlaterNet eps = 10 continued from iteration 33,545, once
with the authors' gradient ("real") and once with the full one ("full"), to
see whether its energy rise after about 35,000 iterations comes from the
term that "real" leaves out.

    python branch_run.py make ORIGINAL STEP OUT --gradient real|full
        OUT, a new folder, gets
          - a copy of ORIGINAL/checkpoint_<STEP>.npz, its only checkpoint, so
            train_moire.py continues from exactly that one;
          - the rows of ORIGINAL/train_stats.csv before STEP;
          - ORIGINAL/settings.json with only "gradient" changed;
          - branch.json: where the branch came from.
        The original folder is only read.
    python branch_run.py compare ORIGINAL BRANCH
        For the iterations from STEP on that both folders have: which rows of
        train_stats.csv are identical, and the largest difference per column.

Then train OUT with train_moire.py and the original's options
(launch_gradient_test.sh does).  It continues from STEP with the saved
weights, KFAC state (its step counter included), walkers, random key, step
width and iteration count.
"""
import argparse
import csv
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

# The columns of train_stats.csv that two identical runs share ("seconds" is wall time).
COMPARED = ["energy", "variance", "energy_imag", "pmove", "step_width", "learning_rate", "fisher_step_sq"]


def md5(path):
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def make(original, step, out, gradient):
    """Make the branch folder `out`.  Stops, writing nothing, if anything is wrong."""
    original, out = Path(original), Path(out)
    checkpoint = original / f"checkpoint_{step:06d}.npz"
    if gradient not in ("real", "full"):
        sys.exit(f"--gradient must be real or full, not {gradient}")
    if out.exists() and any(out.iterdir()):
        sys.exit(f"{out} is not empty; a branch is only made into a new folder")
    try:
        with np.load(checkpoint) as saved:
            done = int(saved["done"])
    except Exception as error:
        sys.exit(f"cannot read {checkpoint} ({error})")
    if done != step:
        sys.exit(f"{checkpoint.name} holds {done} iterations, not {step}")
    settings = json.loads((original / "settings.json").read_text())
    with open(original / "train_stats.csv", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [row for row in reader if int(row["step"]) < step]
        columns = reader.fieldnames
    if [int(row["step"]) for row in rows] != list(range(step)):
        sys.exit(f"{original}/train_stats.csv does not hold exactly one row for each iteration before {step}")

    out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(checkpoint, out / checkpoint.name)
    if md5(out / checkpoint.name) != md5(checkpoint):
        sys.exit(f"the copy of {checkpoint.name} differs from the original")
    with open(out / "train_stats.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)                     # the original's text, unchanged
    (out / "settings.json").write_text(json.dumps(dict(settings, gradient=gradient), indent=2))
    (out / "branch.json").write_text(json.dumps(dict(
        original=str(original.resolve()), step=step, checkpoint=checkpoint.name,
        checkpoint_md5=md5(checkpoint), rows_copied=len(rows),
        changed={"gradient": [settings["gradient"], gradient]},
        made=time.strftime("%Y-%m-%d %H:%M:%S")), indent=2))


def compare(original, branch, log=print):
    """Rows of train_stats.csv from the branch's STEP on, in both folders."""
    original, branch = Path(original), Path(branch)
    start = json.loads((branch / "branch.json").read_text())["step"]

    def rows(folder):
        with open(folder / "train_stats.csv", newline="") as handle:
            return {int(row["step"]): row for row in csv.DictReader(handle) if int(row["step"]) >= start}
    a, b = rows(original), rows(branch)
    common = sorted(set(a) & set(b))
    identical = [step for step in common if all(a[step][c] == b[step][c] for c in COMPARED)]
    first_difference = next((step for step in common if step not in identical), None)
    largest = {c: max((abs(float(a[step][c]) - float(b[step][c])) for step in common), default=None)
               for c in COMPARED}
    at_start = {c: abs(float(a[start][c]) - float(b[start][c])) for c in COMPARED} if start in common else {}
    log(f"iterations {start}-{common[-1] if common else '-'} in both: {len(common)}; "
        f"identical rows: {len(identical)}; first row that differs: {first_difference}")
    log(f"  {'column':15s} {'difference in row ' + str(start):>26s} {'largest, all rows':>22s}")
    for c in COMPARED:
        log(f"  {c:15s} {at_start.get(c, float('nan')):26.6g} {largest[c] if largest[c] is not None else float('nan'):22.6g}")
    return dict(start=start, common=common, identical=identical, first_difference=first_difference,
                largest=largest, at_start=at_start)


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    commands = parser.add_subparsers(dest="command", required=True)
    making = commands.add_parser("make")
    making.add_argument("original", type=Path)
    making.add_argument("step", type=int)
    making.add_argument("out", type=Path)
    making.add_argument("--gradient", required=True)
    comparing = commands.add_parser("compare")
    comparing.add_argument("original", type=Path)
    comparing.add_argument("branch", type=Path)
    args = parser.parse_args()
    if args.command == "make":
        make(args.original, args.step, args.out, args.gradient)
        print(f"made {args.out}")
    else:
        compare(args.original, args.branch)


if __name__ == "__main__":
    main()
