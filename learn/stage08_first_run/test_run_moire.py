"""
Stage 8 checks for run_moire.py.  Run:  python test_run_moire.py   (seconds, no GPU)

A resumed run, and the decision that a run has already finished, both trust
the newest checkpoint that passes checkpoint_problem.  If an incomplete or
damaged file passed, a 21-hour run could be declared finished, or fail when
the authors' code tries to restore it.  And a restart must keep the
iteration target the run was started with: a 1600-iteration run restarted
without --iterations must not continue to 200,000.

The checkpoints here are small stand-ins written like the authors'
checkpoint.save (members t, data, params, opt_state, mcmc_width), so the
tests need no trained run.
"""
import json
import shutil
import tempfile
import zipfile
from pathlib import Path

from absl.testing import absltest
import numpy as np

from run_moire import checkpoint_problem, iteration_target

STEP = 1599


def write_checkpoint(path, step):
    """A small checkpoint with the five members the authors' checkpoint.save writes."""
    np.savez(path, t=np.array(step),
             data=np.array({"positions": np.zeros((1, 8, 12))}, dtype=object),
             params=np.array({"w": np.random.default_rng(0).normal(size=(64, 64))}, dtype=object),
             opt_state=np.array({}, dtype=object),
             mcmc_width=np.array([0.2]))


class CheckpointProblemTest(absltest.TestCase):

    def setUp(self):
        self.folder = Path(tempfile.mkdtemp())
        self.good = self.folder / f"qmcjax_ckpt_{STEP:06d}.npz"
        write_checkpoint(self.good, STEP)

    def tearDown(self):
        shutil.rmtree(self.folder)

    def test_a_complete_checkpoint_passes(self):
        self.assertIsNone(checkpoint_problem(self.good, STEP))

    def test_incomplete_or_damaged_checkpoints_are_rejected(self):
        only_the_step = self.folder / "only_t.npz"
        np.savez(only_the_step, t=np.array(STEP))
        self.assertIn("missing", checkpoint_problem(only_the_step, STEP))

        self.assertIn(f"holds step {STEP}", checkpoint_problem(self.good, STEP - 100))   # wrong name

        truncated = self.folder / "truncated.npz"
        truncated.write_bytes(self.good.read_bytes()[: self.good.stat().st_size // 2])
        self.assertIsNotNone(checkpoint_problem(truncated, STEP))

        # One byte changed inside the stored parameters: the zip directory is
        # intact, so np.load would open the file; only the CRC check sees it.
        damaged = self.folder / "damaged.npz"
        raw = bytearray(self.good.read_bytes())
        with zipfile.ZipFile(self.good) as archive:
            member = archive.getinfo("params.npy")
        raw[member.header_offset + 100 + member.compress_size // 2] ^= 0xFF
        damaged.write_bytes(bytes(raw))
        self.assertIn("damaged member", checkpoint_problem(damaged, STEP))


class IterationTargetTest(absltest.TestCase):

    def test_a_restart_keeps_the_saved_target_unless_asked_otherwise(self):
        folder = Path(tempfile.mkdtemp())
        self.assertIsNone(iteration_target(folder, None))            # new folder: the authors' value
        (folder / "overrides.json").write_text(json.dumps(
            {"authors_command_line": [], "overrides": {"optim.iterations": 1600}}))
        self.assertEqual(iteration_target(folder, None), 1600)        # restarted without --iterations
        self.assertEqual(iteration_target(folder, 5000), 5000)        # explicitly extended
        shutil.rmtree(folder)


if __name__ == "__main__":
    absltest.main()
