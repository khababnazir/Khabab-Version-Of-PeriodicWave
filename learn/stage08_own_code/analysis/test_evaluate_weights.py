"""
Checks for evaluate_weights.py.  Run:  python test_evaluate_weights.py   (CPU, a few minutes)

Each test guards a mistake that would silently change the evaluation:
  parts        the moire and Coulomb parts must be the two potentials inside
               the local energy, or the split kinetic / moire / Coulomb is wrong;
  supercell    the supercell must be the one the archived evaluation of the
               reference run used, or our figures and theirs show different cells;
  weights      the evaluation must read the checkpoint's own weights, walkers
               and step width, and refuse a checkpoint of the other network;
  files        a tiny evaluation writes the three files with matching step
               counts and energy parts that add up; it is then recognised as
               finished, and an evaluation with another seed is not;
  sanity       a wave function whose energy is far from the end of training
               stops the evaluation, and the summary records the failure.
Added for 18 electrons (2026-11-02):
  settings     the network is rebuilt from settings.json exactly as trained
               (Table II sizes, folx, slogdet); old folders get the old values;
  parts 18     the energy split holds at 18 electrons with folx + slogdet;
  files 18     a tiny 18-electron evaluation: 27 sites, no filling patterns,
               156 x 156 maps, every electron counted once;
  checkpoint   --checkpoint evaluates THAT checkpoint, goes to its own folder,
               and the sanity check uses the training just before it (the
               last row of a step when a restart repeated it).
Added after the review of 2026-11-02:
  finished     only a complete evaluation counts as finished (main() skips
               it): every observable there, every number finite, the sanity
               check passed; the nine real evaluations still count as finished.
"""
import csv
import dataclasses
import json
import os
import sys
import tempfile
from pathlib import Path
from unittest import mock

# Before JAX is imported: never on a GPU (a GPU may be training).
os.environ["JAX_PLATFORMS"] = "cpu"
from absl.testing import absltest     # noqa: E402
import numpy as np                     # noqa: E402
import jax                             # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import evaluate_weights                # noqa: E402  (puts train_moire.py and Stages 3-7 on the path)
import train_moire                     # noqa: E402
import slaternet                       # noqa: E402
from local_energy import kinetic_energy  # noqa: E402

# From runs/archive/stage08_authors_code/evaluation/*/summary.json: the supercell of the reference run.
ARCHIVED_SUPERCELL = [[3.0, -1.5], [0.0, 2.598076211353316]]
WALKERS = 8                            # a tiny batch: the tests check bookkeeping, not physics


EIGHTEEN = dataclasses.replace(train_moire.TABLE_II, laplacian="folx", determinants="slogdet")


def fake_run(folder, training_energy, electrons=6):
    """A training folder with a final checkpoint of an untrained SlaterNet (eps = 10).

    At 6 electrons the settings.json of a folder made before 2026-10-30 (no
    "electrons", no "settings"); at 18 the one train_moire.py writes for Table I.
    """
    settings = dict(network="slaternet", eps=10.0, gradient="real", seed=0, iterations=7)
    if electrons == 6:
        problem = train_moire.make_problem("slaternet", 10.0)
    else:
        problem = train_moire.make_problem("slaternet", 10.0, electrons, EIGHTEEN)
        settings.update(electrons=electrons, settings=dataclasses.asdict(EIGHTEEN))
    params = problem.init(jax.random.PRNGKey(1), problem.arch)
    walkers = jax.random.normal(jax.random.PRNGKey(2), (WALKERS, electrons, 2))
    (folder / "settings.json").write_text(json.dumps(settings))
    (folder / "train_stats.csv").write_text("step,energy\n" + "".join(f"{i},{training_energy}\n" for i in range(7)))
    train_moire.save_checkpoint(folder, 7, params, {}, walkers, 0.2, jax.random.PRNGKey(3))
    return problem, params, walkers


class EvaluateWeightsTest(absltest.TestCase):

    def test_parts_are_the_potentials_of_the_local_energy(self):
        R = jax.random.uniform(jax.random.PRNGKey(0), (6, 2), minval=-2.0, maxval=2.0)
        for network in ("slaternet", "attention"):
            for eps in (5.0, 10.0):                   # the Coulomb strength differs
                problem = train_moire.make_problem(network, eps)
                params = problem.init(jax.random.PRNGKey(1), problem.arch)
                moire, coulomb = evaluate_weights.potential_parts(eps)(R)
                kinetic = kinetic_energy(lambda X: problem.log_psi(params, X), R)
                total = problem.local_energy(params, R)
                # float32: the parts are tens of code units, so agreement to ~1e-5 relative
                np.testing.assert_allclose(np.real(total) - moire - coulomb, np.real(kinetic),
                                           rtol=1e-4, atol=1e-3)

    def test_supercell_is_the_one_of_the_archived_evaluation(self):
        np.testing.assert_allclose(slaternet.supercell_vectors(9), ARCHIVED_SUPERCELL, atol=1e-12)

    def test_weights_walkers_and_width_come_from_the_checkpoint(self):
        with tempfile.TemporaryDirectory() as scratch:
            folder = Path(scratch)
            problem, params, walkers = fake_run(folder, training_energy=0.0)
            like = problem.init(jax.random.PRNGKey(99), problem.arch)    # other values, same layout
            done, loaded, loaded_walkers, width = train_moire.load_parameters(
                folder / "checkpoint_000007.npz", like)
            self.assertEqual((done, width), (7, 0.2))
            np.testing.assert_array_equal(loaded_walkers, walkers)
            for a, b in zip(jax.tree.leaves(loaded), jax.tree.leaves(params)):
                np.testing.assert_array_equal(a, b)
            attention = train_moire.make_problem("attention", 10.0)
            with self.assertRaisesRegex(ValueError, "does not fit"):
                train_moire.load_parameters(folder / "checkpoint_000007.npz",
                                            attention.init(jax.random.PRNGKey(0), attention.arch))

    def test_a_tiny_evaluation_writes_consistent_files(self):
        with tempfile.TemporaryDirectory() as scratch:
            run, out = Path(scratch) / "run", Path(scratch) / "out"
            run.mkdir(), out.mkdir()
            fake_run(run, training_energy=0.0)
            summary = evaluate_weights.evaluate(run, out, steps=3, burn_in=1, seed=0,
                                                check_steps=1000, log=lambda *_: None)   # no check
            with open(out / "energies.csv", newline="") as handle:
                rows = list(csv.DictReader(handle))
            with np.load(out / "observables.npz") as totals:
                self.assertEqual(int(totals["steps_done"]), 3)
                self.assertEqual(int(totals["density"].sum()), 3 * WALKERS * 6)   # every electron once
                self.assertEqual(int(totals["patterns"].sum()), 3 * WALKERS)      # every configuration once
            self.assertEqual((len(rows), summary["steps_done"]), (3, 3))
            for row in rows:
                parts = float(row["kinetic"]) + float(row["moire"]) + float(row["coulomb"])
                self.assertAlmostEqual(parts, float(row["energy"]), delta=1e-3 * abs(float(row["energy"])) + 1e-3)

            wanted = dict(checkpoint_md5=summary["checkpoint_md5"], steps_requested=3, burn_in=1, seed=0)
            self.assertEqual(evaluate_weights.existing_evaluation(out, wanted), "finished")
            self.assertEqual(evaluate_weights.existing_evaluation(out, dict(wanted, seed=1)), "other")

    def test_the_sanity_check_stops_a_wrong_wave_function(self):
        with tempfile.TemporaryDirectory() as scratch:
            run, out = Path(scratch) / "run", Path(scratch) / "out"
            run.mkdir(), out.mkdir()
            fake_run(run, training_energy=1000.0)       # nowhere near this network's energy
            with self.assertRaisesRegex(SystemExit, "sanity check FAILED"):
                evaluate_weights.evaluate(run, out, steps=3, burn_in=1, seed=0,
                                          check_steps=2, log=lambda *_: None)
            check = json.loads((out / "summary.json").read_text())["sanity_check"]
            self.assertFalse(check["passed"])

    # ------------------------------------------------- added for 18 electrons

    def test_settings_are_rebuilt_from_settings_json(self):
        with tempfile.TemporaryDirectory() as scratch:
            folder = Path(scratch)
            fake_run(folder, 0.0, electrons=18)
            saved, settings = evaluate_weights.run_settings(folder)
            self.assertEqual((saved["electrons"], settings), (18, EIGHTEEN))
            fake_run(folder, 0.0)                               # a folder made before 2026-10-30
            saved, settings = evaluate_weights.run_settings(folder)
            self.assertEqual(saved["electrons"], 6)
            self.assertEqual(settings, dataclasses.replace(train_moire.Settings(),
                                                           **train_moire.BEFORE_18_ELECTRONS))
            text = json.loads((folder / "settings.json").read_text())
            text["settings"] = {"no_such_setting": 1}
            (folder / "settings.json").write_text(json.dumps(text))
            with self.assertRaisesRegex(SystemExit, "no_such_setting"):
                evaluate_weights.run_settings(folder)

    def test_only_a_complete_finite_checked_evaluation_counts_as_finished(self):
        """main() skips an evaluation that existing_evaluation calls finished, so
        a missing observable, a number that is not finite, or a failed or
        missing sanity check must make it 'other'."""
        with tempfile.TemporaryDirectory() as scratch:
            run, out = Path(scratch) / "run", Path(scratch) / "out"
            run.mkdir(), out.mkdir()
            fake_run(run, training_energy=0.0)
            summary = evaluate_weights.evaluate(run, out, steps=3, burn_in=1, seed=0,
                                                check_steps=1000, log=lambda *_: None)   # no check
            wanted = dict(checkpoint_md5=summary["checkpoint_md5"], steps_requested=3, burn_in=1, seed=0)
            good = {name: (out / name).read_bytes() for name in ("summary.json", "energies.csv", "observables.npz")}
            with np.load(out / "observables.npz") as saved:
                arrays = {name: saved[name] for name in saved.files}

            def verdict(new_summary=None, csv_text=None, npz=None):
                for name, content in good.items():
                    (out / name).write_bytes(content)
                if new_summary is not None:
                    (out / "summary.json").write_text(json.dumps(new_summary))
                if csv_text is not None:
                    (out / "energies.csv").write_text(csv_text)
                if npz is not None:
                    with open(out / "observables.npz", "wb") as handle:
                        np.savez(handle, **npz)
                return evaluate_weights.existing_evaluation(out, wanted)

            self.assertEqual(verdict(), "finished")
            for name in evaluate_weights.OBSERVABLE_ARRAYS + ("patterns", "steps_done"):   # 9 sites: patterns too
                with self.subTest(missing=name):
                    self.assertEqual(verdict(npz={k: v for k, v in arrays.items() if k != name}), "other")
            inexact = [k for k, v in arrays.items() if np.issubdtype(v.dtype, np.inexact)]
            self.assertNotEmpty(inexact)
            for name in inexact:
                with self.subTest(not_finite=name):
                    bad = arrays[name].copy()
                    bad.flat[0] = np.nan
                    self.assertEqual(verdict(npz=dict(arrays, **{name: bad})), "other")
            lines = good["energies.csv"].decode().splitlines()
            header, first = lines[0].split(","), lines[1].split(",")
            first[header.index("energy")] = "nan"
            self.assertEqual(verdict(csv_text="\n".join([lines[0], ",".join(first)] + lines[2:]) + "\n"), "other")
            self.assertEqual(verdict(new_summary=dict(summary, sanity_check=dict(passed=False))), "other")
            self.assertEqual(verdict(new_summary=dict(summary, sanity_check=dict(passed=True))), "finished")
            with mock.patch.object(evaluate_weights, "CHECK_STEPS", 3):     # then 3 steps must have been checked
                self.assertEqual(verdict(), "other")

    def test_the_real_evaluations_still_count_as_finished(self):
        """The stricter check must not turn the existing evaluations into 'other'."""
        summaries = sorted((train_moire.REPO / "runs" / "stage08").glob("*/evaluation/*/summary.json"))
        self.assertGreaterEqual(len(summaries), 9)
        for path in summaries:
            saved = json.loads(path.read_text())
            wanted = {name: saved[name] for name in ("checkpoint_md5", "steps_requested", "burn_in", "seed")}
            with self.subTest(path.parent.name):
                self.assertEqual(evaluate_weights.existing_evaluation(path.parent, wanted), "finished")

    def test_parts_are_the_potentials_at_18_electrons(self):
        R = jax.random.uniform(jax.random.PRNGKey(0), (18, 2), minval=-3.0, maxval=3.0)
        moire, coulomb = evaluate_weights.potential_parts(5.0, electrons=18)(R)
        for network in ("slaternet", "attention"):
            problem = train_moire.make_problem(network, 5.0, 18, EIGHTEEN)
            params = problem.init(jax.random.PRNGKey(1), problem.arch)
            kinetic = kinetic_energy(lambda X: problem.log_psi(params, X), R)   # Stage 4's Hessian
            total = problem.local_energy(params, R)                             # folx + slogdet
            # float32 and two different routes to the kinetic energy: ~1e-4 relative.
            # A wrong cell (9 sites instead of 27) would miss by whole code units.
            np.testing.assert_allclose(np.real(total) - moire - coulomb, np.real(kinetic),
                                       rtol=1e-3, atol=1e-2)

    def test_a_tiny_18_electron_evaluation(self):
        with tempfile.TemporaryDirectory() as scratch:
            run, out = Path(scratch) / "training" / "run", Path(scratch) / "out"
            run.mkdir(parents=True), out.mkdir()
            fake_run(run, training_energy=0.0, electrons=18)
            summary = evaluate_weights.evaluate(run, out, steps=2, burn_in=1, seed=0,
                                                check_steps=1000, log=lambda *_: None)
            self.assertEqual((summary["n_electrons"], summary["n_sites"], summary["density_bins"]),
                             (18, 27, 156))
            with np.load(out / "observables.npz") as totals:
                self.assertNotIn("patterns", totals.files)
                self.assertEqual(totals["density"].shape, (156, 156))
                self.assertEqual(int(totals["density"].sum()), 2 * WALKERS * 18)   # every electron once
                self.assertEqual(int(totals["occupation"].sum()), 2 * WALKERS * 18)
            with open(out / "energies.csv", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertIn("occupation_26", rows[0])
            self.assertNotIn("occupation_27", rows[0])
            for row in rows:
                parts = float(row["kinetic"]) + float(row["moire"]) + float(row["coulomb"])
                self.assertAlmostEqual(parts, float(row["energy"]), delta=1e-3 * abs(float(row["energy"])) + 1e-2)
            saved, _ = evaluate_weights.run_settings(run)
            self.assertEqual(evaluate_weights.default_out(run, saved), Path(scratch) / "evaluation" / "run")

    def test_an_earlier_checkpoint_is_evaluated_as_asked(self):
        with tempfile.TemporaryDirectory() as scratch:
            run, out = Path(scratch) / "training" / "run", Path(scratch) / "out"
            run.mkdir(parents=True), out.mkdir()
            problem, _, walkers = fake_run(run, training_energy=0.0)
            early = problem.init(jax.random.PRNGKey(5), problem.arch)
            train_moire.save_checkpoint(run, 3, early, {}, walkers, 0.3, jax.random.PRNGKey(6))
            summary = evaluate_weights.evaluate(run, out, steps=2, burn_in=1, seed=0, checkpoint=3,
                                                check_steps=1000, log=lambda *_: None)
            self.assertEqual((summary["checkpoint"], summary["checkpoint_step"], summary["final_checkpoint"],
                              summary["mcmc_width"]), ("checkpoint_000003.npz", 3, False, 0.3))
            self.assertEqual(summary["checkpoint_md5"], evaluate_weights.file_md5(run / "checkpoint_000003.npz"))
            # The sanity check must compare with the training BEFORE checkpoint 3 (rows 0-2,
            # mean 1001), not with the end of the run (rows 0-6, mean 1003).  Far from this
            # network's energy, so it fails -- and the summary records what it compared.
            (run / "train_stats.csv").write_text("step,energy\n" + "".join(f"{i},{1000 + i}\n" for i in range(7)))
            with self.assertRaisesRegex(SystemExit, "sanity check FAILED"):
                evaluate_weights.evaluate(run, out, steps=2, burn_in=1, seed=0, checkpoint=3,
                                          check_steps=1, log=lambda *_: None)
            check = json.loads((out / "summary.json").read_text())["sanity_check"]
            self.assertEqual(check["training_last_steps"], 1001.0)
            saved, _ = evaluate_weights.run_settings(run)
            self.assertEqual(evaluate_weights.default_out(run, saved, 3).name, "run_at_000003")
            self.assertEqual(evaluate_weights.default_out(run, saved, 7).name, "run")
            self.assertEqual(evaluate_weights.default_out(run, saved), Path(scratch) / "evaluation" / "run")
            with self.assertRaisesRegex(SystemExit, "not inside a training/ folder"):
                evaluate_weights.default_out(Path(scratch) / "elsewhere", saved)

    def test_the_sanity_check_uses_the_training_just_before_the_checkpoint(self):
        stats = np.array([[i, float(i)] for i in range(5000)])
        np.testing.assert_array_equal(evaluate_weights.training_window(stats, 3000), np.arange(1000, 3000))
        # A restart from checkpoint 2500 repeated steps 2500 ... 2999: their LAST rows count.
        repeated = np.vstack([stats[:3000], [[i, -1.0] for i in range(2500, 3000)]])
        np.testing.assert_array_equal(evaluate_weights.training_window(repeated, 3000),
                                      np.r_[np.arange(1000, 2500), -np.ones(500)])
        with self.assertRaisesRegex(SystemExit, "does not hold"):
            evaluate_weights.training_window(stats[:2500], 3000)


if __name__ == "__main__":
    jax.config.update("jax_default_matmul_precision", "float32")     # as the evaluation
    absltest.main()
