"""
Our code with the HONEYCOMB FIELD: learn/stage08_own_code/train_moire.py, unchanged, run with its
local energy plus the same field as in the authors'-code test
(authors_exact_code_runs/archive/eps3_seed42_honeycomb_field/honeycomb_field.py):

    E_field(r_1 .. r_18) = lambda * sum_i sum_{j=1,2,3} cos(kappa_j . r_i)

kappa_j the three corners of the moire Brillouin zone at 0, 120, 240 degrees (|kappa| = 4 pi / 3,
a_M = 1).  On the moire sites it is 3 lambda on sublattice 0 (the one with the origin) and
-1.5 lambda on the other two: sublattice 0 raised by 4.5 lambda.  lambda = 0.5 code units = 1.69 meV.

    CUDA_VISIBLE_DEVICES=G ../.venv/bin/python field_training.py LAMBDA <train_moire.py options>

wraps train_moire.make_problem so that the local energy it returns includes the field, then runs
train_moire.main with the options.  Everything else -- settings.json, checkpoints, train_stats.csv
(whose energies include the field while it is on) -- is train_moire's.  The field is switched off
by continuing the same folder with plain train_moire.py and a larger --iterations; a note,
field.json, records in the folder which iterations had the field.
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "learn" / "stage08_own_code"))
import train_moire                                    # noqa: E402  unchanged

KAPPAS = (4 * np.pi / 3) * np.array([[np.cos(a), np.sin(a)] for a in (0.0, 2 * np.pi / 3, 4 * np.pi / 3)])


def field_energy(R, strength):
    """lambda * sum over electrons and the three kappas of cos(kappa . r); R of shape (N, 2)."""
    import jax.numpy as jnp
    return strength * jnp.sum(jnp.cos(R @ jnp.asarray(KAPPAS, dtype=R.dtype).T))


def with_field(make_problem, strength):
    """make_problem, but the Problem's local energy includes the field."""
    def make(*args, **kwargs):
        problem = make_problem(*args, **kwargs)
        plain = problem.local_energy
        return problem._replace(local_energy=lambda params, R: plain(params, R) + field_energy(R, strength))
    return make


def main():
    strength = float(sys.argv[1])
    sys.argv = [str(REPO / "learn" / "stage08_own_code" / "train_moire.py")] + sys.argv[2:]
    out = Path(sys.argv[sys.argv.index("--out") + 1]).resolve()
    iterations = int(sys.argv[sys.argv.index("--iterations") + 1])
    out.mkdir(parents=True, exist_ok=True)
    note = out / "field.json"
    if note.exists() and json.loads(note.read_text()) != dict(strength=strength, iterations_with_field=f"0 to {iterations - 1}"):
        sys.exit(f"{note} records another field; not continuing")
    note.write_text(json.dumps(dict(strength=strength, iterations_with_field=f"0 to {iterations - 1}"), indent=2))
    train_moire.make_problem = with_field(train_moire.make_problem, strength)
    train_moire.main()


if __name__ == "__main__":
    main()
