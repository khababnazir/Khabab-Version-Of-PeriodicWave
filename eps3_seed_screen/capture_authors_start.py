"""
The authors' seed-42 START, taken from their own train.py: the initial parameters (network.init)
and the initial walker positions (init_electrons_gaussian), exactly as their train.py makes them
for the eps = 3 seed-42 run that found the honeycomb
(authors_exact_code_runs/runs/18_electrons_27_sites_eps3_seed42/ordinary/), before any burn-in or
training.

How: their seed-42 script (run_2deg_triangular_potential_seed42.py, unchanged) is run in a
temporary folder (so no checkpoint is found and their train.py starts fresh), with two functions
wrapped: the network's init records the parameters it returns, and their init_electrons_gaussian
records the positions it returns and then stops the run (raises Captured), before anything is
trained.  Their seed is 42 (cfg.debug.deterministic), their keys are split as in their train.py.

Run, from this folder (CPU; a few seconds after JAX has started):
    JAX_PLATFORMS=cpu ../.venv/bin/python capture_authors_start.py authors_seed42_start.pkl
Output: a pickle with params (their pytree, numpy arrays), positions (1024, 36), spins, and the
script and seed it came from.
"""
import os
import pickle
import runpy
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SEED42_SCRIPT = REPO / "authors_exact_code_runs/runs/18_electrons_27_sites_eps3_seed42/run_2deg_triangular_potential_seed42.py"
ARGS = ["18_0", "27", "0.35", "0.3333333333333333", "8.031", "15", "45", "CustomPsiformer"]


class Captured(Exception):
    """Raised once the start is recorded, to stop their train.py before any training."""


def capture():
    import attr
    import jax
    from periodicwave import CustomPsiformer, train

    got = {}
    real_make_fermi_net = CustomPsiformer.make_fermi_net
    real_init_electrons = train.init_electrons_gaussian

    def make_fermi_net(*args, **kwargs):
        network = real_make_fermi_net(*args, **kwargs)
        real_init = network.init

        def init(key):
            params = real_init(key)
            assert "params" not in got, "network.init called twice"
            got["params"] = jax.tree.map(np.asarray, jax.device_get(params))
            return params
        return attr.evolve(network, init=init)

    def init_electrons_gaussian(*args, **kwargs):
        positions, spins = real_init_electrons(*args, **kwargs)
        assert "params" in got, "the walkers came before the parameters: not their train.py order"
        got["positions"], got["spins"] = np.asarray(positions), np.asarray(spins)
        raise Captured

    CustomPsiformer.make_fermi_net, train.init_electrons_gaussian = make_fermi_net, init_electrons_gaussian
    saved_argv, saved_cwd = sys.argv, os.getcwd()
    try:
        with tempfile.TemporaryDirectory() as scratch:
            os.chdir(scratch)
            sys.argv = [str(SEED42_SCRIPT)] + ARGS
            try:
                runpy.run_path(str(SEED42_SCRIPT), run_name="__main__")
            except Captured:
                pass
            else:
                raise RuntimeError("their script ended without making walkers: nothing captured")
    finally:
        sys.argv = saved_argv
        os.chdir(saved_cwd)
        CustomPsiformer.make_fermi_net, train.init_electrons_gaussian = real_make_fermi_net, real_init_electrons
    return dict(got, script=str(SEED42_SCRIPT.relative_to(REPO)), arguments=ARGS, seed=42,
                backend=jax.default_backend())


def main():
    out = Path(sys.argv[1])
    start = capture()
    assert start["positions"].shape == (1024, 36), start["positions"].shape
    out.write_bytes(pickle.dumps(start))
    print(f"captured: {len(jax_leaves(start['params']))} parameter arrays, walkers {start['positions'].shape}, "
          f"backend {start['backend']} -> {out}")


def jax_leaves(tree):
    import jax
    return jax.tree.leaves(tree)


if __name__ == "__main__":
    main()
