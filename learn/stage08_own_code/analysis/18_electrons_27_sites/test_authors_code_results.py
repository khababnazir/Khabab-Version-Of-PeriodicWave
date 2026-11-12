"""
Tests for authors_code_results.py: files written the way the authors' code writes them, with
contents whose answer is known by hand.

    JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= uv run --no-project --python ../../../../.venv/bin/python \\
        --with matplotlib --with pyblock python test_authors_code_results.py
"""
import json
import tempfile
from pathlib import Path

import numpy as np
from absl.testing import absltest

import authors_code_results as acr
import checkpoint_structure as cs

GEO = cs.geometry()
SUPERCELL, SITES, _, SUBLATTICE = GEO


def honeycomb_walkers(n_walkers, seed=0):
    """Walkers that are honeycombs (sublattice 0 empty), each electron a little off its site."""
    rng = np.random.default_rng(seed)
    filled = SITES[SUBLATTICE != 0]                                  # 18 sites
    return np.stack([rng.permutation(filled) for _ in range(n_walkers)]) + 0.02 * rng.normal(size=(n_walkers, 18, 2))


def write_checkpoint(path, R, electron_major=True):
    """A checkpoint in their format: data["positions"] = (walkers, 36), x1, y1, x2, y2, ..."""
    flat = R.reshape(len(R), 36) if electron_major else R.transpose(0, 2, 1).reshape(len(R), 36)
    np.savez(path, t=7, data=dict(positions=flat.astype(np.float32), spins=None), mcmc_width=np.ones(1))


def write_run(folder, energies, iterations, scale=acr.U_AT_EPS_5):
    """A run folder in their format: config.json and train_stats.csv."""
    folder.mkdir(parents=True)
    config = dict(batch_size=1024, optim=dict(iterations=iterations, lr=dict(rate=0.1)),
                  network=dict(CustomPsiformer=dict(num_layers=4, num_heads=4)),
                  system=dict(make_local_energy_kwargs=dict(potential_kwargs=dict(interaction_energy_scale=scale))))
    (folder / "config.json").write_text(json.dumps(config))
    rows = ["step,energy,ewmean,ewvar,pmove,locstd"] + [f"{i},{e},0,0,0.5,1" for i, e in enumerate(energies)]
    (folder / "train_stats.csv").write_text("\n".join(rows) + "\n")


class WalkersTest(absltest.TestCase):

    def test_a_honeycomb_in_their_file_format_is_seen_as_a_honeycomb(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "qmcjax_ckpt_000123.npz"
            write_checkpoint(path, honeycomb_walkers(64))
            R = acr.walkers(path, SUPERCELL)
            self.assertEqual(R.shape, (64, 18, 2))
            measured = cs.measure(R, *GEO)
            self.assertAlmostEqual(measured["S_sites"], 4.5, places=9)
            self.assertGreater(measured["S_kappa"], 4.3)             # 0.02 a_M of spread costs a little
            self.assertAlmostEqual(measured["sublattice_0"], 0.0, places=9)
            self.assertAlmostEqual(measured["sublattice_1"], 0.5, places=9)

    def test_a_file_in_the_other_layout_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "qmcjax_ckpt_000123.npz"
            write_checkpoint(path, honeycomb_walkers(64), electron_major=False)
            with self.assertRaisesRegex(AssertionError, "scrambled"):
                acr.walkers(path, SUPERCELL)

    def test_checkpoints_come_in_order_of_iteration(self):
        with tempfile.TemporaryDirectory() as tmp:
            for step in (2000, 30, 100000):
                write_checkpoint(Path(tmp) / f"qmcjax_ckpt_{step:06d}.npz", honeycomb_walkers(8, seed=step))
            self.assertEqual([i for i, _ in acr.checkpoint_rows(Path(tmp), GEO)], [30, 2000, 100000])


class EnergyTest(absltest.TestCase):

    def test_energy_is_the_mean_of_the_last_fifth_in_meV_per_electron(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "run"
            energies = np.concatenate([np.full(800, -100.0), np.full(200, -300.0)])   # last 20 %: -300
            write_run(run, energies, iterations=1000)
            result = acr.training_energy(run, 1000)
            self.assertAlmostEqual(result["energy"], -300.0 * 3.375559 / 18, places=9)
            self.assertEqual(result["rows_in_window"], 200)
            self.assertTrue(result["finished"])
            self.assertEqual(result["last_step"], 999)

    def test_an_unfinished_run_is_marked_and_a_half_written_row_is_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "run"
            write_run(run, np.full(900, -100.0), iterations=1000)
            with open(run / "train_stats.csv", "a") as handle:
                handle.write("900,-1")                               # no end of line: still being written
            result = acr.training_energy(run, 1000)
            self.assertFalse(result["finished"])
            self.assertEqual(result["last_step"], 899)
            self.assertEqual(result["rows_in_window"], 100)          # steps 800 .. 899
            self.assertAlmostEqual(result["energy"], -100.0 * 3.375559 / 18, places=9)

    def test_steps_that_are_not_consecutive_stop_the_script(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "run"
            write_run(run, np.full(10, -100.0), iterations=10)
            with open(run / "train_stats.csv", "a") as handle:
                handle.write("5,-100,0,0,0.5,1\n")                   # a restart appended an old step
            with self.assertRaisesRegex(AssertionError, "steps"):
                acr.training_energy(run, 10)

    def test_eps_is_read_from_the_interaction_strength(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_run(Path(tmp) / "a", [0.0], 10, scale=10.62347336201578)
            write_run(Path(tmp) / "b", [0.0], 10, scale=5.31173668100789)
            self.assertEqual(acr.settings(Path(tmp) / "a")["eps"], 5.0)
            self.assertEqual(acr.settings(Path(tmp) / "b")["eps"], 10.0)
            self.assertIn(5.0, acr.PAPER)
            self.assertIn(10.0, acr.PAPER)


class WholeRunTest(absltest.TestCase):

    def test_only_checkpoints_of_the_last_fifth_are_pooled(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "results" / "x" / "y" / "z"
            write_run(run, np.full(1000, -100.0), iterations=1000)
            even = np.concatenate([SITES[SUBLATTICE == g][:6] for g in range(3)])   # 6, 6, 6: S_sites 0
            write_checkpoint(run / "qmcjax_ckpt_000500.npz", np.repeat(even[None], 8, axis=0))
            write_checkpoint(run / "qmcjax_ckpt_000900.npz", honeycomb_walkers(16))
            self.assertEqual(acr.their_runs(Path(tmp)), [run])
            row, rows = acr.analyse(run, GEO)
            self.assertEqual([r["iteration"] for r in rows], [500, 900])
            self.assertEqual((row["pooled_checkpoints"], row["pooled_walkers"]), (1, 16))
            self.assertAlmostEqual(row["pooled_S_sites"], 4.5, places=9)   # the even checkpoint is not in it
            self.assertAlmostEqual(rows[0]["S_sites"], 0.0, places=9)
            self.assertAlmostEqual(row["minus_paper"], -100.0 * 3.375559 / 18 + 59.127, places=9)


if __name__ == "__main__":
    absltest.main()
