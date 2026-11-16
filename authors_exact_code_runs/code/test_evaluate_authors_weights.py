"""
Checks for evaluate_authors_weights.py.  Run, from this folder (CPU, about a minute):
    ../../.venv/bin/python test_evaluate_authors_weights.py

  configuration  for all four runs, running the script each was started with (theirs, or the
                 Table II copy) with the arguments of how_this_was_started.sh rebuilds exactly the
                 config.json their code wrote;
  refusals       the wrong script (theirs for the Table II run, the copy for the released run) or
                 the wrong 1/eps is refused -- the evaluation cannot pick up another configuration;
  output         each run and 1/eps finds exactly one output folder, the one config.json names;
  steps          by default steps x walkers = 20,480,000, as in our evaluations;
  their code     external/PeriodicWave is at the authors' commit with no tracked file changed;
  checkpoint     each final checkpoint restores with the authors' own checkpoint.restore, holds the
                 last iteration, and its parameters fit the network built from the configuration.

What this cannot show: that the evaluation itself is right.  The evaluation steps are the
archived evaluator's, unchanged (it was checked against the training energies of the 6-electron
runs); each run checks itself against the end of its training after 200 steps (sanity check).
"""
import json
import os
import sys
from pathlib import Path

os.environ["JAX_PLATFORMS"] = "cpu"           # before JAX is imported: a GPU may be busy
from absl.testing import absltest
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import evaluate_authors_weights as ev         # noqa: E402

RUNS = HERE.parent / "runs"
RELEASED = RUNS / "18_electrons_27_sites_released_settings"
TABLE2 = RUNS / "18_electrons_27_sites_table2_settings"
CASES = [(RELEASED, 0.2), (RELEASED, 0.1), (TABLE2, 0.2), (TABLE2, 0.1)]


class ConfigurationTest(absltest.TestCase):

    def test_each_run_rebuilds_its_config_json(self):
        for run, inverse_eps in CASES:
            with self.subTest(run=run.name, inverse_eps=inverse_eps):
                cfg, trained_with = ev.training_configuration(run, inverse_eps)     # exits if different
                self.assertEqual(trained_with["authors_command_line"][3], str(inverse_eps))
                expected = TABLE2 / ev.TABLE2_SCRIPT if run == TABLE2 else ev.AUTHORS_SCRIPT
                self.assertEqual(ev.authors_script(run), expected)
                self.assertEqual(cfg.batch_size, 4096 if run == TABLE2 else 1024)

    def test_the_wrong_script_is_refused(self):
        for run, wrong in ((TABLE2, ev.AUTHORS_SCRIPT), (RELEASED, TABLE2 / ev.TABLE2_SCRIPT)):
            original = ev.authors_script
            ev.authors_script = lambda folder, wrong=wrong: wrong          # noqa: E731
            try:
                with self.subTest(run=run.name), self.assertRaisesRegex(SystemExit, "differs"):
                    ev.training_configuration(run, 0.2)
            finally:
                ev.authors_script = original

    def test_the_wrong_eps_is_refused(self):
        original = ev.command_line
        ev.command_line = lambda inverse_eps: original(0.1 if inverse_eps == 0.2 else 0.2)   # noqa: E731
        try:
            with self.assertRaisesRegex(SystemExit, "differs"):
                ev.training_configuration(RELEASED, 0.2)
        finally:
            ev.command_line = original


class FoldersTest(absltest.TestCase):

    def test_one_output_folder_each_named_in_config_json(self):
        for run, inverse_eps in CASES:
            folder = ev.output_folder(run, inverse_eps)
            recorded = json.loads((folder / "config.json").read_text())
            self.assertEqual(run / recorded["log"]["save_path"], folder)
            scale = recorded["system"]["make_local_energy_kwargs"]["potential_kwargs"]["interaction_energy_scale"]
            np.testing.assert_allclose(scale, 53.11736681007890 * inverse_eps, rtol=1e-12)

    def test_default_steps_give_our_number_of_configurations(self):
        for run, inverse_eps in CASES:
            batch = json.loads((ev.output_folder(run, inverse_eps) / "config.json").read_text())["batch_size"]
            self.assertEqual((ev.CONFIGURATIONS // batch) * batch, 20_480_000)

    def test_the_authors_code_is_unchanged(self):
        ev.authors_code_unchanged()                                       # exits if not


class CheckpointTest(absltest.TestCase):

    def test_final_checkpoints_fit_the_network(self):
        import jax
        import jax.numpy as jnp
        from periodicwave import checkpoint
        for run, inverse_eps in CASES:
            with self.subTest(run=run.name, inverse_eps=inverse_eps):
                cfg, _ = ev.training_configuration(run, inverse_eps)
                network, _, _ = ev.build_like_train_py(cfg)
                final = ev.output_folder(run, inverse_eps) / f"qmcjax_ckpt_{cfg.optim.iterations - 1:06d}.npz"
                completed, data, params, _, width = checkpoint.restore(str(final), cfg.batch_size)
                self.assertEqual(completed, cfg.optim.iterations)
                params = jax.tree_util.tree_map(lambda x: jnp.asarray(x)[0], params)
                self.assertTrue(ev.same_structure(params, network.init(jax.random.PRNGKey(0))))
                self.assertEqual(np.shape(data.positions)[-2:], (cfg.batch_size, 36))
                self.assertTrue(np.all(np.isfinite(np.asarray(data.positions))))


if __name__ == "__main__":
    absltest.main()
