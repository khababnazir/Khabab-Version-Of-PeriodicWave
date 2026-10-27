"""
Stage 8 check for run_moire.py's checkpoint validation.  Run:  python test_run_moire.py   (seconds, no GPU)

A resumed run, and the decision that a run has already finished, both trust
the newest checkpoint that passes checkpoint_problem.  If an incomplete or
damaged file passed, a 21-hour run could be declared finished, or fail when
the authors' code tries to restore it.  The good example is a real
checkpoint of the short smoke run (runs/stage08/smoke).
"""
import shutil
import tempfile
import zipfile
from pathlib import Path

from absl.testing import absltest
import numpy as np

from run_moire import checkpoint_problem

SMOKE_CHECKPOINT = (Path(__file__).resolve().parents[2] / "runs" / "stage08" / "smoke"
                    / "psiformer_eps5" / "qmcjax_ckpt_001599.npz")


class CheckpointProblemTest(absltest.TestCase):

    def setUp(self):
        if not SMOKE_CHECKPOINT.exists():
            self.skipTest(f"needs {SMOKE_CHECKPOINT}")
        self.folder = Path(tempfile.mkdtemp())
        self.good = self.folder / SMOKE_CHECKPOINT.name
        shutil.copy(SMOKE_CHECKPOINT, self.good)

    def tearDown(self):
        shutil.rmtree(self.folder)

    def test_a_complete_checkpoint_passes(self):
        self.assertIsNone(checkpoint_problem(self.good, 1599))

    def test_incomplete_or_damaged_checkpoints_are_rejected(self):
        only_the_step = self.folder / "only_t.npz"
        np.savez(only_the_step, t=np.array(1599))
        self.assertIn("missing", checkpoint_problem(only_the_step, 1599))

        self.assertIn("holds step 1599", checkpoint_problem(self.good, 1499))   # wrong name

        truncated = self.folder / "truncated.npz"
        truncated.write_bytes(self.good.read_bytes()[: self.good.stat().st_size // 2])
        self.assertIsNotNone(checkpoint_problem(truncated, 1599))

        # One byte changed inside the stored parameters: the zip directory is
        # intact, so np.load would open the file; only the CRC check sees it.
        damaged = self.folder / "damaged.npz"
        raw = bytearray(self.good.read_bytes())
        with zipfile.ZipFile(self.good) as archive:
            member = archive.getinfo("params.npy")
        middle = member.header_offset + 100 + member.compress_size // 2
        raw[middle] ^= 0xFF
        damaged.write_bytes(bytes(raw))
        self.assertIn("damaged member", checkpoint_problem(damaged, 1599))


if __name__ == "__main__":
    absltest.main()
