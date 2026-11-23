"""
Checks for evaluate_authors_weights.py.  Run, from this folder (CPU, about a minute):
    ../../.venv/bin/python test_evaluate_authors_weights.py

  configuration  for all six runs (eps = 5 and 10 with both settings; eps = 3, self-attention and
                 SlaterNet, added 2026-11-19), running the script each was started with (theirs, or the
                 Table II copy) with the arguments of how_this_was_started.sh rebuilds exactly the
                 config.json their code wrote;
  refusals       the wrong script (theirs for the Table II run, the copy for the released run) or
                 the wrong 1/eps is refused -- the evaluation cannot pick up another configuration;
  output         each run, 1/eps and network finds exactly one output folder, the one config.json
                 names; the wrong network is refused; the two networks of the eps = 3 folder get
                 different evaluation folders, and the earlier runs keep theirs;
  steps          by default steps x walkers = 20,480,000, as in our evaluations;
  seed 42        (added 2026-11-23) the two seed-42 eps = 3 run folders (ordinary/ and
                 honeycomb_field/phase2_field_off/) find the seed-42 script above them and rebuild
                 their config.json; phase 1 (the field) is refused, because its config.json names the
                 field; train_stats.csv must hold consecutive iterations ending at the checkpoint
                 (phase 2: 20,000 to 199,999, after its copied checkpoint 19,999) -- a gap, a repeated
                 row, a missing end or a late start without that checkpoint is refused;
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
EPS3 = RUNS / "18_electrons_27_sites_eps3_released_settings"
SEED42 = RUNS / "18_electrons_27_sites_eps3_seed42"
ORDINARY = SEED42 / "ordinary"
PHASE1 = SEED42 / "honeycomb_field" / "phase1_field_on"
PHASE2 = SEED42 / "honeycomb_field" / "phase2_field_off"
CASES = [(RELEASED, 0.2, "CustomPsiformer"), (RELEASED, 0.1, "CustomPsiformer"),
         (TABLE2, 0.2, "CustomPsiformer"), (TABLE2, 0.1, "CustomPsiformer"),
         (EPS3, 1 / 3, "CustomPsiformer"), (EPS3, 1 / 3, "SlaterNet"),
         (ORDINARY, 1 / 3, "CustomPsiformer"), (PHASE2, 1 / 3, "CustomPsiformer")]


def expected_script(run):
    if run == TABLE2:
        return TABLE2 / ev.TABLE2_SCRIPT
    if run in (ORDINARY, PHASE2):
        return SEED42 / ev.SEED42_SCRIPT
    return ev.AUTHORS_SCRIPT


class ConfigurationTest(absltest.TestCase):

    def test_each_run_rebuilds_its_config_json(self):
        for run, inverse_eps, network in CASES:
            with self.subTest(run=run.name, inverse_eps=inverse_eps, network=network):
                cfg, trained_with = ev.training_configuration(run, inverse_eps, network)     # exits if different
                self.assertEqual(trained_with["authors_command_line"][3], str(inverse_eps))
                self.assertEqual(ev.authors_script(run), expected_script(run))
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
        ev.command_line = lambda inverse_eps, network="CustomPsiformer": original(          # noqa: E731
            0.1 if inverse_eps == 0.2 else 0.2, network)
        try:
            with self.assertRaisesRegex(SystemExit, "differs"):
                ev.training_configuration(RELEASED, 0.2)
        finally:
            ev.command_line = original

    def test_the_wrong_network_is_refused(self):
        """The two eps = 3 runs: rebuilding one with the other network's arguments must fail."""
        original = ev.command_line
        swap = {"CustomPsiformer": "SlaterNet", "SlaterNet": "CustomPsiformer"}
        ev.command_line = lambda inverse_eps, network="CustomPsiformer": original(inverse_eps, swap[network])  # noqa: E731
        try:
            for network in ev.NETWORKS:
                with self.subTest(network=network), self.assertRaisesRegex(SystemExit, "differs"):
                    ev.training_configuration(EPS3, 1 / 3, network)
        finally:
            ev.command_line = original

    def test_one_third_is_the_argument_the_runs_were_started_with(self):
        self.assertEqual(ev.command_line(ev.INVERSE_EPS["1/3"], "SlaterNet")[3], "0.3333333333333333")
        start = (EPS3 / "how_this_was_started.sh").read_text()
        self.assertIn("INVERSE_EPS=0.3333333333333333", start)


    def test_seed42_phase1_is_refused(self):
        """Phase 1 ran with the field; the seed-42 script without the field cannot rebuild its config."""
        self.assertEqual(ev.authors_script(PHASE1), SEED42 / ev.SEED42_SCRIPT)
        with self.assertRaisesRegex(SystemExit, "differs"):
            ev.training_configuration(PHASE1, 1 / 3)


class TrainingRowsTest(absltest.TestCase):

    def test_the_two_seed42_runs(self):
        self.assertLen(ev.training_energies(ev.output_folder(ORDINARY, 1 / 3), 200_000), 200_000)
        self.assertLen(ev.training_energies(ev.output_folder(PHASE2, 1 / 3), 200_000), 180_000)

    def fake(self, steps, checkpoints=()):
        folder = Path(self.create_tempdir().full_path)
        lines = ["step,energy,ewmean,ewvar,pmove,locstd"] + [f"{s},-1.0,-1.0,0.0,0.5,1.0" for s in steps]
        (folder / "train_stats.csv").write_text("\n".join(lines) + "\n")
        for c in checkpoints:
            (folder / f"qmcjax_ckpt_{c:06d}.npz").write_bytes(b"")
        return folder

    def test_complete_rows_are_accepted(self):
        self.assertLen(ev.training_energies(self.fake(range(100)), 100), 100)
        self.assertLen(ev.training_energies(self.fake(range(20, 100), checkpoints=[19]), 100), 80)

    def test_bad_rows_are_refused(self):
        cases = {"a gap": list(range(50)) + list(range(51, 100)),
                 "a repeated row (a restart)": list(range(60)) + list(range(55, 100)),
                 "a missing end": range(99),
                 "an empty table": []}
        for name, steps in cases.items():
            with self.subTest(name), self.assertRaisesRegex(SystemExit, "consecutive"):
                ev.training_energies(self.fake(steps), 100)
        with self.assertRaisesRegex(SystemExit, "continued from"):
            ev.training_energies(self.fake(range(20, 100)), 100)       # no qmcjax_ckpt_000019.npz


class FoldersTest(absltest.TestCase):

    def test_evaluation_folders(self):
        """Earlier runs keep their evaluation folder; the eps = 3 networks get different ones."""
        released = ev.output_folder(RELEASED, 0.2)
        self.assertEqual(ev.default_out(RELEASED, released, "CustomPsiformer"), RELEASED / "evaluation" / released.name)
        outs = {network: ev.default_out(EPS3, ev.output_folder(EPS3, 1 / 3, network), network) for network in ev.NETWORKS}
        self.assertNotEqual(outs["CustomPsiformer"], outs["SlaterNet"])
        self.assertEqual(outs["SlaterNet"].name, "SlaterNet_el18_0_N27_V4.44370908_45.0_U17.70578894")

    def test_one_output_folder_each_named_in_config_json(self):
        for run, inverse_eps, network in CASES:
            folder = ev.output_folder(run, inverse_eps, network)
            recorded = json.loads((folder / "config.json").read_text())
            self.assertEqual(run / recorded["log"]["save_path"], folder)
            scale = recorded["system"]["make_local_energy_kwargs"]["potential_kwargs"]["interaction_energy_scale"]
            np.testing.assert_allclose(scale, 53.11736681007890 * inverse_eps, rtol=1e-12)

    def test_default_steps_give_our_number_of_configurations(self):
        for run, inverse_eps, network in CASES:
            batch = json.loads((ev.output_folder(run, inverse_eps, network) / "config.json").read_text())["batch_size"]
            self.assertEqual((ev.CONFIGURATIONS // batch) * batch, 20_480_000)

    def test_the_authors_code_is_unchanged(self):
        ev.authors_code_unchanged()                                       # exits if not


class CheckpointTest(absltest.TestCase):

    def test_final_checkpoints_fit_the_network(self):
        import jax
        import jax.numpy as jnp
        from periodicwave import checkpoint
        for run, inverse_eps, network in CASES:
            with self.subTest(run=run.name, inverse_eps=inverse_eps, network=network):
                cfg, _ = ev.training_configuration(run, inverse_eps, network)
                wave_function, _, _ = ev.build_like_train_py(cfg)
                final = ev.output_folder(run, inverse_eps, network) / f"qmcjax_ckpt_{cfg.optim.iterations - 1:06d}.npz"
                completed, data, params, _, width = checkpoint.restore(str(final), cfg.batch_size)
                self.assertEqual(completed, cfg.optim.iterations)
                params = jax.tree_util.tree_map(lambda x: jnp.asarray(x)[0], params)
                self.assertTrue(ev.same_structure(params, wave_function.init(jax.random.PRNGKey(0))))
                self.assertEqual(cfg.network.network_type, network)
                self.assertEqual(np.shape(data.positions)[-2:], (cfg.batch_size, 36))
                self.assertTrue(np.all(np.isfinite(np.asarray(data.positions))))


if __name__ == "__main__":
    absltest.main()
