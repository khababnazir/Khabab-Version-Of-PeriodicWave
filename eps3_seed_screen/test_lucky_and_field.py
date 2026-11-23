"""
Checks for the lucky-start and field runs of our code.  Run, from this folder (CPU, a few minutes):
    JAX_PLATFORMS=cpu $HOME/.local/bin/uv run --no-project --python ../.venv/bin/python --with matplotlib \
        --with pyblock python test_lucky_and_field.py

  capture     capture_authors_start.py gives the same start twice (bit for bit); its walkers are
              the authors' Gaussian start (1024 x 18 electrons, mean 0, spread their init_width);
  converter   lucky_start's copy of ours_from_authors_psiformer is, line for line, the function
              in learn/stage08_own_code/test_networks.py;
  network     our network with the converted seed-42 parameters gives the authors' network's
              log|Psi| and phase at 16 configurations spread over the supercell (their starting
              walkers are crowded around the origin, where the phase is too sensitive to
              single-precision rounding for a sharp test);
  checkpoint  the iteration-0 checkpoint written for our train_moire.py reads back with
              train_moire.load_checkpoint: done 0, these parameters, these walkers, width 0.2, and
              the key a fresh start with that seed has after drawing its parameters and walkers;
              the walkers are their starting positions in their layout x1, y1, x2, y2, ... (in
              single precision, as both codes compute);
  field       field_training's field is 3 lambda on every site of sublattice 0 and -1.5 lambda on
              the others, periodic in the supercell; the wrapped local energy is the plain one
              plus exactly the field, and with lambda = 0 the plain one.
"""
import ast
import os
import sys
import tempfile
from pathlib import Path

os.environ["JAX_PLATFORMS"] = "cpu"
from absl.testing import absltest                     # noqa: E402
import numpy as np                                    # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "authors_exact_code_runs" / "code"))
sys.path.insert(0, str(REPO / "learn" / "stage08_own_code" / "analysis" / "18_electrons_27_sites"))
import capture_authors_start as cap                   # noqa: E402
import lucky_start                                    # noqa: E402
import field_training                                 # noqa: E402
import evaluate_authors_weights as ev                 # noqa: E402
import checkpoint_structure as cs                     # noqa: E402

import jax                                            # noqa: E402
import jax.numpy as jnp                               # noqa: E402

START = cap.capture()


class CaptureTest(absltest.TestCase):

    def test_the_same_start_twice(self):
        again = cap.capture()
        np.testing.assert_array_equal(again["positions"], START["positions"])
        for a, b in zip(jax.tree.leaves(again["params"]), jax.tree.leaves(START["params"])):
            np.testing.assert_array_equal(a, b)

    def test_walkers_are_their_gaussian_start(self):
        cfg = ev.authors_configuration(REPO / cap.SEED42_SCRIPT.relative_to(REPO), cap.ARGS)
        x = START["positions"]
        self.assertEqual(x.shape, (1024, 36))
        self.assertLess(abs(x.mean()), 0.01)
        np.testing.assert_allclose(x.std(), cfg.mcmc.init_width, rtol=0.02)


class ConverterTest(absltest.TestCase):

    def test_copy_is_learns_function(self):
        def function_source(path, name):
            tree = ast.parse(Path(path).read_text())
            node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
            node.body = [s for s in node.body if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))]
            return ast.dump(node)
        learn = REPO / "learn" / "stage08_own_code" / "test_networks.py"
        self.assertEqual(function_source(lucky_start.__file__, "ours_from_authors_psiformer"),
                         function_source(learn, "ours_from_authors_psiformer"))

    def test_converted_network_gives_theirs(self):
        cfg = ev.authors_configuration(REPO / cap.SEED42_SCRIPT.relative_to(REPO), cap.ARGS)
        theirs, _, _ = ev.build_like_train_py(cfg)
        problem = lucky_start.train_moire.make_problem("attention", 3.0, 18, lucky_start.SETTINGS)
        with tempfile.TemporaryDirectory() as scratch:
            path = Path(scratch) / "start.pkl"
            path.write_bytes(__import__("pickle").dumps(START))
            ours_params, ours_walkers = lucky_start.converted_start(path)
        their_params = jax.tree.map(jnp.asarray, START["params"])
        supercell, _, _, _ = cs.geometry()
        u = np.random.default_rng(2).uniform(size=(16, 18, 2))
        R = jnp.asarray(u @ np.asarray(supercell).T, dtype=jnp.float32)          # spread over the cell
        phase, log_abs = jax.vmap(lambda r: theirs.apply(their_params, r.reshape(-1), jnp.ones(18),
                                                         jnp.zeros((1, 2)), jnp.array([0.0])))(R)
        ours = jax.vmap(lambda r: problem.log_psi(ours_params, r))(R)
        self.assertEqual(ours_walkers.shape, (1024, 18, 2))
        np.testing.assert_allclose(ours.log_abs, log_abs, rtol=1e-4, atol=1e-3)
        np.testing.assert_allclose(np.exp(1j * (ours.phase - phase)), 1.0, atol=1e-3)


class CheckpointTest(absltest.TestCase):

    def test_iteration_zero_checkpoint_reads_back(self):
        tm = lucky_start.train_moire
        with tempfile.TemporaryDirectory() as scratch:
            folder, start = Path(scratch), Path(scratch) / "start.pkl"
            start.write_bytes(__import__("pickle").dumps(START))
            params, opt_state, walkers, width, key = lucky_start.first_checkpoint(start, 7)
            tm.save_checkpoint(folder, 0, params, opt_state, walkers, width, key)
            done, p, o, w, wd, k = tm.load_checkpoint(folder / "checkpoint_000000.npz", params, opt_state)
        self.assertEqual(done, 0)
        for a, b in zip(jax.tree.leaves(p), jax.tree.leaves(params)):
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
        np.testing.assert_array_equal(np.asarray(w), np.asarray(walkers))
        np.testing.assert_array_equal(np.asarray(w).reshape(1024, 36), START["positions"].astype(np.float32))
        self.assertEqual(float(wd), 0.2)
        expected_key = jax.random.split(jax.random.PRNGKey(7), 3)[0]
        np.testing.assert_array_equal(np.asarray(k), np.asarray(expected_key))


class FieldTest(absltest.TestCase):

    def test_values_on_the_sites(self):
        supercell, sites, _, sublattice = cs.geometry()
        values = np.array([float(field_training.field_energy(jnp.asarray(s[None]), 0.5)) for s in sites])
        np.testing.assert_allclose(values[sublattice == 0], 1.5, atol=1e-5)
        np.testing.assert_allclose(values[sublattice != 0], -0.75, atol=1e-5)
        R = jnp.asarray(np.random.default_rng(0).uniform(-3, 3, (18, 2)))
        for shift in np.asarray(supercell).T:
            np.testing.assert_allclose(field_training.field_energy(R + shift, 0.5),
                                       field_training.field_energy(R, 0.5), atol=1e-4)

    def test_wrapped_local_energy_adds_exactly_the_field(self):
        tm = lucky_start.train_moire
        plain = tm.make_problem("attention", 3.0, 18, lucky_start.SETTINGS)
        params = plain.init(jax.random.PRNGKey(3), plain.arch)
        R = jnp.asarray(np.random.default_rng(1).normal(size=(18, 2)), dtype=jnp.float32)
        for strength in (0.5, 0.0):
            wrapped = field_training.with_field(tm.make_problem, strength)("attention", 3.0, 18, lucky_start.SETTINGS)
            difference = wrapped.local_energy(params, R) - plain.local_energy(params, R)
            np.testing.assert_allclose(complex(difference), float(field_training.field_energy(R, strength)), atol=1e-3)


if __name__ == "__main__":
    absltest.main()
