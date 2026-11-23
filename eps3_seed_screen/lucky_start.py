"""
Our code from the authors' LUCKY START: their seed-42 initial parameters and walkers (captured by
capture_authors_start.py), given to OUR train_moire.py (learn/stage08_own_code, unchanged) as an
iteration-0 checkpoint.

    ../.venv/bin/python lucky_start.py START.pkl FOLDER SEED

writes FOLDER/checkpoint_000000.npz with train_moire.save_checkpoint: done = 0, our parameters
converted from theirs, their 1024 starting walkers, the step width of a fresh start (0.2), a fresh
optimiser state for these parameters, and the random key a fresh start with --seed SEED would
have after drawing its parameters and walkers.  Then train_moire.py ... --seed SEED --out FOLDER
finds the checkpoint, and because it holds iteration 0 it runs its usual burn-in (300 x 30 moves)
and trains, exactly as a fresh start with seed SEED, except that parameters and walkers are the
authors' seed-42 ones.  SEED therefore only chooses our Metropolis moves and KFAC's random numbers;
those cannot be the authors' (each code draws them in its own way).

The settings are the released ones at eps = 3, 18 electrons (folx, slogdet), as the screen.
ours_from_authors_psiformer is a copy of learn/stage08_own_code/test_networks.py's (that module
switches JAX to 64-bit when imported, so it is not imported here); test_lucky_and_field.py checks
the copy against it and the converted network against theirs.
"""
import dataclasses
import json
import pickle
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
CODE = REPO / "learn" / "stage08_own_code"
sys.path.insert(0, str(CODE))
for stage in ("stage03_slater", "stage04_mcmc", "stage05_attention", "stage06_kfac", "stage07_ewald"):
    sys.path.insert(0, str(REPO / "learn" / stage))
import train_moire                                    # noqa: E402  unchanged

EPS, ELECTRONS, GRADIENT = 3.0, 18, "real"
SETTINGS = dataclasses.replace(train_moire.SETTINGS["released"], laplacian="folx", determinants="slogdet")


def ours_from_authors_psiformer(p):
    """Copy of learn/stage08_own_code/test_networks.py's (checked equal in test_lucky_and_field.py)."""
    layers = p["layers"]
    return {
        "embed": {"w": layers["embed"]},
        "layers": [{"attention": {"queries": {"w": attention["q_w"]},
                                  "keys": {"w": attention["k_w"]},
                                  "values": {"w": attention["v_w"]},
                                  "output": {"w": attention["attn_output"]}},
                    "attention_norm": dict(ln[0]),
                    "perceptrons": [{"w": m["w"], "b": m["b"]} for m in mlp],
                    "perceptron_norm": dict(ln[1])}
                   for attention, ln, mlp in zip(layers["attention"], layers["ln"], layers["mlp"])],
        "orbital": {"w": p["orbital"][0]["w"]},
    }


def converted_start(start_file):
    """(our parameters, our walkers (1024, 18, 2)) from the captured start."""
    import jax.numpy as jnp
    start = pickle.loads(Path(start_file).read_bytes())
    params = ours_from_authors_psiformer(start["params"])
    params = __import__("jax").tree.map(lambda x: jnp.asarray(x, dtype=jnp.float32), params)
    walkers = jnp.asarray(start["positions"], dtype=jnp.float32).reshape(-1, ELECTRONS, 2)   # x1, y1, x2, y2, ...
    return params, walkers


def first_checkpoint(start_file, seed):
    """What train_moire.train would hold before its burn-in, with the authors' parameters and walkers."""
    import jax
    import jax.numpy as jnp
    problem = train_moire.make_problem("attention", EPS, ELECTRONS, SETTINGS)
    optimizer = train_moire.make_optimizer(train_moire.make_loss(problem, GRADIENT), SETTINGS)
    key = jax.random.PRNGKey(seed)                     # as train_moire.train: one key each for parameters,
    key, key_params, _ = jax.random.split(key, 3)      # walkers, everything after
    ours = problem.init(key_params, problem.arch)      # only to check the structure
    params, walkers = converted_start(start_file)
    assert jax.tree.structure(params) == jax.tree.structure(ours), "converted parameters: other structure"
    assert all(a.shape == b.shape for a, b in zip(jax.tree.leaves(params), jax.tree.leaves(ours)))
    assert walkers.shape == (SETTINGS.batch, ELECTRONS, 2)
    opt_state = optimizer.init(params, key, (walkers, jnp.zeros(SETTINGS.batch)))
    return params, opt_state, walkers, SETTINGS.move_width, key


def main():
    start_file, folder, seed = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
    folder.mkdir(parents=True, exist_ok=True)
    if list(folder.glob("checkpoint_*.npz")):
        sys.exit(f"{folder} already holds checkpoints; not writing a start over them")
    import jax
    jax.config.update("jax_default_matmul_precision", "float32")     # as train_moire.main
    params, opt_state, walkers, width, key = first_checkpoint(start_file, seed)
    train_moire.save_checkpoint(folder, 0, params, opt_state, walkers, width, key)
    (folder / "lucky_start.json").write_text(json.dumps(dict(
        start=str(start_file), seed_of_our_moves=seed,
        what="parameters and walkers of the authors' seed-42 start; our burn-in, moves and KFAC from seed_of_our_moves"),
        indent=2))
    print(f"wrote {folder / 'checkpoint_000000.npz'}")


if __name__ == "__main__":
    main()
