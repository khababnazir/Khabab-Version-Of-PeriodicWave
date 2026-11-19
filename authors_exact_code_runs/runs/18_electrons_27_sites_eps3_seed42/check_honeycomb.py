"""
Did a honeycomb form?  S(kappa) and the three sublattice shares of the walkers in one checkpoint of
the authors' code, measured with learn's functions, unchanged (authors_code_results.walkers, with
its layout guard; checkpoint_structure.measure).  A honeycomb with sublattice 0 empty (the one the
field raises) gives S(kappa) = 4.5 and shares 0, 1/2, 1/2; an even liquid about 0.7 and 1/3 each.

    $HOME/.local/bin/uv run --no-project --python ../../../.venv/bin/python --with matplotlib --with pyblock \
        python check_honeycomb.py CHECKPOINT [OUT.csv]
"""
import csv
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "learn" / "stage08_own_code" / "analysis" / "18_electrons_27_sites"))
import authors_code_results as acr                    # noqa: E402  learn's, unchanged

cs = acr.cs


def main():
    path = Path(sys.argv[1]).resolve()
    geo = cs.geometry()
    cs.self_check(geo)
    R = acr.walkers(path, geo[0])
    row = dict(checkpoint=path.name, walkers=len(R), **cs.measure(R, *geo))
    print(f"{path.name}: {len(R)} walkers | S(kappa) {row['S_kappa']:.3f} (connected {row['S_kappa_connected']:.3f}) | "
          f"sublattice shares {row['sublattice_0']:.3f} {row['sublattice_1']:.3f} {row['sublattice_2']:.3f} | "
          f"walkers with at most 3 electrons on their emptiest sublattice: {row['emptiest_at_most_3']:.3f}")
    if len(sys.argv) > 2:
        out = Path(sys.argv[2])
        new = not out.exists()
        with open(out, "a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            if new:
                writer.writeheader()
            writer.writerow(row)


if __name__ == "__main__":
    main()
