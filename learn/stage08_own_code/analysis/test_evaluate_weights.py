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
"""
import csv
import json
import os
import sys
import tempfile
from pathlib import Path

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


def fake_run(folder, training_energy):
    """A training folder with a final checkpoint of an untrained SlaterNet (eps = 10)."""
    problem = train_moire.make_problem("slaternet", 10.0)
    params = problem.init(jax.random.PRNGKey(1), problem.arch)
    walkers = jax.random.normal(jax.random.PRNGKey(2), (WALKERS, 6, 2))
    settings = dict(network="slaternet", eps=10.0, gradient="real", seed=0, iterations=7)
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


if __name__ == "__main__":
    jax.config.update("jax_default_matmul_precision", "float32")     # as the evaluation
    absltest.main()
