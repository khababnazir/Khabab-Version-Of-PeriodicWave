"""
PARADOX EXPERIMENT -- checks for the twisted boundary conditions.
Run, from this folder:   ../../.venv/bin/python test_twist.py      (CPU, double precision, a few minutes)

  twist vector   k_twist . L_i = 2 pi T_i for both supercell vectors (6 and 18 electrons);
  boundary       moving ANY one electron by L_i leaves |Psi| unchanged and multiplies Psi by
                 exp(2 pi i T_i) -- on the real networks (attention at 6 and 18 electrons,
                 SlaterNet), for unequal twists, a twist along one direction only, and the
                 experiment's (1/3, 1/3);
  zero twist     without a twist this copy IS the original: log Psi and the local energy are
                 identical number for number to learn/stage08_own_code/train_moire.py, and
                 a small training run writes the same checkpoint and the same train_stats.csv;
  kinetic        a determinant of plane waves with the twist has kinetic energy
                 1/2 sum_j |k_j + k_twist|^2 at every configuration, through both of Stage 4's
                 routes (full Hessian and folx) -- an exact answer, not a comparison with
                 our own formula;
  local energy   on the network, built by make_problem: twisted minus periodic local energy
                 equals 1/2 N |k|^2 + k . sum grad(phase) - i k . sum grad(log|Psi|), and
                 log|Psi| is unchanged (so the sampling is unchanged);
  settings       the launcher's exact command line stores the twist in the run's settings;
                 a folder with another twist is refused; a folder made by the original code
                 counts as periodic;
  evaluation     the evaluate_weights.py of this folder imports THIS folder's train_moire
                 and gives a twisted run its twist;
  shells         twist_shells.py agrees with learn's shell_structure.py and with the trainer's
                 constants; the periodic shells are those of the diagnosis; (1/3, 1/3) is closed.

What this cannot show: what the trained network does.  That is the experiment.
"""
import csv
import dataclasses
import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path
from unittest import mock

os.environ["JAX_PLATFORMS"] = "cpu"           # before JAX is imported: a GPU may be training
from absl.testing import absltest
import jax
jax.config.update("jax_enable_x64", True)     # before any JAX operation
import numpy as np
import jax.numpy as jnp

HERE = Path(__file__).resolve().parent
LEARN = HERE.parents[1] / "learn" / "stage08_own_code"
sys.path.insert(0, str(HERE))
import train_moire as tm                      # THIS folder's copy, with the twist
import twist_shells
import slaternet
from local_energy import (kinetic_energy, kinetic_energy_folx, plane_wave_k_vectors,
                          plane_wave_log_psi, moire_reciprocal_vectors)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


original = load("train_moire_original", LEARN / "train_moire.py")     # the untouched original
THIRD = float(1 / 3)


def twisted(settings, twist_1, twist_2):
    return dataclasses.replace(settings, twist_1=twist_1, twist_2=twist_2)


def table2(module):
    return dataclasses.replace(module.TABLE_II, laplacian="folx", determinants="slogdet")


def some_positions(sites, electrons, seed, n=1):
    supercell = jnp.asarray(slaternet.supercell_vectors(sites))        # columns L1, L2
    return jax.random.uniform(jax.random.PRNGKey(seed), (n, electrons, 2)) @ supercell.T


class TwistVectorTest(absltest.TestCase):

    def test_this_is_the_copy(self):
        self.assertEqual(Path(tm.__file__).resolve().parent, HERE)
        self.assertEqual(Path(original.__file__).resolve().parent, LEARN)

    def test_k_twist_dot_L_is_the_twist_angle(self):
        for sites in (9, 27):
            supercell = np.asarray(slaternet.supercell_vectors(sites))
            G = slaternet.reciprocal_vectors(supercell)
            k = tm.twist_vector(G, 1 / 3, 2 / 7)
            np.testing.assert_allclose(k @ supercell[:, 0], 2 * np.pi / 3, rtol=1e-12)
            np.testing.assert_allclose(k @ supercell[:, 1], 2 * np.pi * 2 / 7, rtol=1e-12)

    def test_the_experiments_twist_vector(self):
        G = slaternet.reciprocal_vectors(np.asarray(slaternet.supercell_vectors(27)))
        k = tm.twist_vector(G, THIRD, THIRD)
        np.testing.assert_allclose(k, [0.698132, 0.403067], atol=1e-6)      # (G1 + G2) / 3
        np.testing.assert_allclose(k, twist_shells.twist_vector(THIRD, THIRD), rtol=1e-12)
        np.testing.assert_allclose(0.5 * 18 * k @ k, 5.8487, atol=1e-3)      # the energy shift 1/2 N |k|^2


class BoundaryTest(absltest.TestCase):
    """Psi(..., r_j + L_i, ...) = exp(2 pi i T_i) Psi(..., r_j, ...), for every electron j."""

    def check(self, network, electrons, settings, twist):
        sites = tm.SITES[electrons]
        problem = tm.make_problem(network, 5.0, electrons, twisted(settings, *twist))
        params = problem.init(jax.random.PRNGKey(1), problem.arch)
        supercell = np.asarray(slaternet.supercell_vectors(sites))
        R = some_positions(sites, electrons, seed=2)[0]
        psi = jax.jit(lambda X: problem.log_psi(params, X))
        before = psi(R)
        for electron in range(electrons):
            for i in (0, 1):
                after = psi(R.at[electron].add(jnp.asarray(supercell[:, i])))
                with self.subTest(network=network, electrons=electrons, twist=twist, electron=electron, L=i + 1):
                    np.testing.assert_allclose(after.log_abs, before.log_abs, rtol=0, atol=1e-8)
                    np.testing.assert_allclose(np.exp(1j * (after.phase - before.phase)),
                                               np.exp(2j * np.pi * twist[i]), rtol=0, atol=1e-7)

    def test_attention_6_electrons(self):
        for twist in ((1 / 3, 2 / 7), (0.5, 0.0), (0.0, 0.25), (0.0, 0.0)):
            self.check("attention", 6, tm.Settings(), twist)

    def test_slaternet_6_electrons(self):
        self.check("slaternet", 6, tm.Settings(), (1 / 3, 2 / 7))

    def test_attention_18_electrons_the_experiment(self):
        self.check("attention", 18, table2(tm), (THIRD, THIRD))
        self.check("attention", 18, table2(tm), (1 / 3, 2 / 7))


class ZeroTwistTest(absltest.TestCase):
    """Without a twist, this copy computes exactly what the original computes."""

    def test_wave_function_and_local_energy_are_identical(self):
        cases = (("attention", 6, {}), ("slaternet", 6, {}),
                 ("attention", 18, dict(laplacian="folx", determinants="slogdet")))
        for network, electrons, changes in cases:
            base_new = tm.TABLE_II if electrons == 18 else tm.Settings()
            base_old = original.TABLE_II if electrons == 18 else original.Settings()
            new = tm.make_problem(network, 5.0, electrons, dataclasses.replace(base_new, **changes))
            old = original.make_problem(network, 5.0, electrons, dataclasses.replace(base_old, **changes))
            params = old.init(jax.random.PRNGKey(3), old.arch)
            for R in some_positions(tm.SITES[electrons], electrons, seed=4, n=2):
                with self.subTest(network=network, electrons=electrons):
                    a, b = new.log_psi(params, R), old.log_psi(params, R)
                    np.testing.assert_array_equal(a.log_abs, b.log_abs)
                    np.testing.assert_array_equal(a.phase, b.phase)
                    np.testing.assert_array_equal(new.local_energy(params, R), old.local_energy(params, R))

    def test_a_small_training_is_identical_and_a_twist_changes_it(self):
        quiet = lambda *args: None                              # noqa: E731
        small = dict(batch=16, burn_in=2, mcmc_steps=3)

        def rows(folder):
            with open(folder / "train_stats.csv", newline="") as handle:
                return [{k: v for k, v in row.items() if k != "seconds"} for row in csv.DictReader(handle)]

        with tempfile.TemporaryDirectory() as scratch:
            old, new, with_twist = Path(scratch, "old"), Path(scratch, "new"), Path(scratch, "twist")
            for folder in (old, new, with_twist):
                folder.mkdir()
            original.train(old, "attention", 5.0, "full", 0, 3, dataclasses.replace(original.Settings(), **small), log=quiet)
            tm.train(new, "attention", 5.0, "full", 0, 3, dataclasses.replace(tm.Settings(), **small), log=quiet)
            tm.train(with_twist, "attention", 5.0, "full", 0, 3,
                     dataclasses.replace(tm.Settings(), twist_1=THIRD, twist_2=THIRD, **small), log=quiet)
            with np.load(old / "checkpoint_000003.npz") as a, np.load(new / "checkpoint_000003.npz") as b:
                self.assertEqual(sorted(a.files), sorted(b.files))
                for name in a.files:
                    np.testing.assert_array_equal(a[name], b[name], err_msg=name)
            self.assertEqual(rows(old), rows(new))
            # The twisted run: the same walkers at step 0 (the sampling does not see the phase),
            # another energy, and finite numbers throughout.
            plain, twist = rows(new), rows(with_twist)
            self.assertEqual(plain[0]["pmove"], twist[0]["pmove"])
            self.assertNotEqual(plain[0]["energy"], twist[0]["energy"])
            self.assertTrue(all(np.isfinite(float(row["energy"])) for row in twist))


class KineticEnergyTest(absltest.TestCase):
    """An exact answer: plane waves exp(i (k_j + k_twist) . r) have kinetic energy 1/2 sum |k_j + k_twist|^2."""

    def test_twisted_plane_waves(self):
        for sites, electrons in ((9, 6), (27, 18)):
            supercell = np.asarray(slaternet.supercell_vectors(sites))
            G = slaternet.reciprocal_vectors(supercell)
            k_vectors = plane_wave_k_vectors(supercell, electrons)
            for twist in ((THIRD, THIRD), (1 / 3, 2 / 7)):
                k_twist = tm.twist_vector(G, *twist)
                exact = 0.5 * np.sum((k_vectors + k_twist) ** 2)
                wrong_sign = 0.5 * np.sum((k_vectors - k_twist) ** 2)
                no_twist = 0.5 * np.sum(k_vectors ** 2)
                self.assertGreater(abs(exact - wrong_sign), 0.05)      # the test can tell a sign mistake
                self.assertGreater(abs(exact - no_twist), 0.05)        # ... and a missing twist
                psi = lambda X: tm.add_twist(plane_wave_log_psi(X, k_vectors), X, k_twist)   # noqa: E731
                for R in some_positions(sites, electrons, seed=5, n=2):
                    for name, kinetic in (("hessian", kinetic_energy), ("folx", kinetic_energy_folx)):
                        with self.subTest(electrons=electrons, twist=twist, route=name):
                            value = complex(kinetic(psi, R))
                            np.testing.assert_allclose(value.real, exact, rtol=1e-9)
                            np.testing.assert_allclose(value.imag, 0.0, atol=1e-8 * exact)


def exact_attention(q, k, v, *args, **kwargs):
    """jax.nn.dot_product_attention with its softmax written out, in the input precision;
    JAX's own computes it in single precision, which would hide mistakes below about 1e-4.
    Copied from learn/stage08_own_code/test_train_moire.py.  Shapes (N, heads, d)."""
    assert not args and not kwargs
    weights = jax.nn.softmax(jnp.einsum("tnh,snh->nts", q, k) / jnp.sqrt(q.shape[-1]), axis=-1)
    return jnp.einsum("nts,snh->tnh", weights, v)


class LocalEnergyTest(absltest.TestCase):
    """The twist reaches the local energy that the training and the evaluation use."""

    def test_twisted_minus_periodic_local_energy(self):
        self.enter_context(mock.patch.object(jax.nn, "dot_product_attention", exact_attention))
        cases = (("attention", 18, table2(tm), (THIRD, THIRD)),
                 ("attention", 6, tm.Settings(), (1 / 3, 2 / 7)),
                 ("slaternet", 6, tm.Settings(), (0.5, 0.0)))
        for network, electrons, settings, twist in cases:
            sites = tm.SITES[electrons]
            periodic = tm.make_problem(network, 5.0, electrons, settings)
            with_twist = tm.make_problem(network, 5.0, electrons, twisted(settings, *twist))
            params = periodic.init(jax.random.PRNGKey(7), periodic.arch)
            G = slaternet.reciprocal_vectors(np.asarray(slaternet.supercell_vectors(sites)))
            k = jnp.asarray(tm.twist_vector(G, *twist))
            for R in some_positions(sites, electrons, seed=8, n=2):
                grad_log_abs = jax.grad(lambda X: periodic.log_psi(params, X).log_abs)(R)     # (N, 2)
                grad_phase = jax.grad(lambda X: periodic.log_psi(params, X).phase)(R)
                expected = (0.5 * electrons * (k @ k) + jnp.sum(grad_phase @ k)
                            - 1j * jnp.sum(grad_log_abs @ k))
                difference = with_twist.local_energy(params, R) - periodic.local_energy(params, R)
                scale = abs(complex(periodic.local_energy(params, R)))
                with self.subTest(network=network, electrons=electrons, twist=twist):
                    self.assertGreater(abs(complex(expected)), 0.1)
                    np.testing.assert_allclose(complex(difference), complex(expected), rtol=0, atol=1e-8 * scale)
                    np.testing.assert_array_equal(with_twist.log_psi(params, R).log_abs,
                                                  periodic.log_psi(params, R).log_abs)


class Stop(Exception):
    pass


class SettingsTest(absltest.TestCase):

    def wanted_from_command_line(self, extra):
        """What main() would store in settings.json for the launcher's command line + extra."""
        launcher = (HERE / "launch_twisted.sh").read_text()
        line = next(line for line in launcher.splitlines() if line.startswith("OPTIONS="))
        options = line.split('"')[1].split()
        if extra is not None:                                      # extra replaces the launcher's twist
            i = options.index("--twist")
            options = options[:i] + options[i + 3:] + extra
        seen = {}

        def capture(folder, wanted, iterations_given):
            seen.update(wanted)
            raise Stop
        with tempfile.TemporaryDirectory() as scratch, \
                mock.patch.object(sys, "argv", ["train_moire.py", *options, "--out", scratch]), \
                mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"}), \
                mock.patch.object(tm, "hold_lock", lambda *args: None), \
                mock.patch.object(jax, "default_backend", lambda: "gpu"), \
                mock.patch.object(tm, "same_experiment_or_stop", capture):
            with self.assertRaises(Stop):
                tm.main()
        return seen

    def test_the_launchers_command_line(self):
        wanted = self.wanted_from_command_line(None)
        self.assertEqual({k: v for k, v in wanted.items() if k != "settings"},
                         dict(network="attention", eps=5.0, electrons=18, gradient="full", seed=0, iterations=150000))
        self.assertEqual(wanted["settings"], dataclasses.asdict(twisted(table2(tm), THIRD, THIRD)))
        self.assertEqual(wanted["settings"]["twist_1"], 1 / 3)
        # Everything except the twist is what the ORIGINAL trainer stores for the zero-twist baseline.
        without = {k: v for k, v in wanted["settings"].items() if k not in ("twist_1", "twist_2")}
        self.assertEqual(without, dataclasses.asdict(table2(original)))

    def test_no_twist_option_means_periodic(self):
        for extra in ([], ["--twist", "0", "0"]):
            settings = self.wanted_from_command_line(extra)["settings"]
            self.assertEqual((settings["twist_1"], settings["twist_2"]), (0.0, 0.0))
        settings = self.wanted_from_command_line(["--twist", "1/2", "0"])["settings"]
        self.assertEqual((settings["twist_1"], settings["twist_2"]), (0.5, 0.0))

    def test_another_twist_is_another_experiment(self):
        def wanted(twist_1, twist_2):
            return dict(network="attention", eps=5.0, electrons=18, gradient="full", seed=0, iterations=150000,
                        settings=dataclasses.asdict(twisted(table2(tm), twist_1, twist_2)))
        with tempfile.TemporaryDirectory() as scratch:
            folder = Path(scratch)
            self.assertEqual(tm.same_experiment_or_stop(folder, wanted(THIRD, THIRD), None), 150000)
            self.assertEqual(json.loads((folder / "settings.json").read_text())["settings"]["twist_1"], THIRD)
            self.assertEqual(tm.same_experiment_or_stop(folder, wanted(THIRD, THIRD), None), 150000)   # a restart
            for other in ((0.0, 0.0), (THIRD, 0.0), (0.0, THIRD), (2 / 3, 2 / 3)):
                with self.assertRaisesRegex(SystemExit, "different experiment"):
                    tm.same_experiment_or_stop(folder, wanted(*other), None)

    def test_a_folder_of_the_original_code_is_periodic(self):
        old = dict(network="attention", eps=5.0, electrons=18, gradient="full", seed=0, iterations=150000,
                   settings=dataclasses.asdict(table2(original)))
        self.assertNotIn("twist_1", old["settings"])
        new = dict(old, settings=dataclasses.asdict(table2(tm)))
        with tempfile.TemporaryDirectory() as scratch:
            folder = Path(scratch)
            (folder / "settings.json").write_text(json.dumps(old))
            self.assertEqual(tm.same_experiment_or_stop(folder, new, None), 150000)
            with self.assertRaisesRegex(SystemExit, "different experiment"):
                tm.same_experiment_or_stop(folder, dict(new, settings=dataclasses.asdict(
                    twisted(table2(tm), THIRD, THIRD))), None)


class EvaluationTest(absltest.TestCase):

    def test_the_evaluation_uses_this_folders_trainer_and_the_runs_twist(self):
        # In a fresh Python, as when the launcher runs it: which train_moire does it import?
        import subprocess
        code = ("import sys; sys.path.insert(0, sys.argv[1]); import evaluate_weights; "
                "print(evaluate_weights.__file__); print(evaluate_weights.train_moire.__file__)")
        out = subprocess.run([sys.executable, "-c", code, str(HERE)], capture_output=True, text=True,
                             env=dict(os.environ, JAX_PLATFORMS="cpu"), check=True).stdout.split()
        self.assertEqual([Path(p).resolve() for p in out], [HERE / "evaluate_weights.py", HERE / "train_moire.py"])
        import evaluate_weights
        self.assertEqual(Path(evaluate_weights.__file__).resolve().parent, HERE)
        import observables
        self.assertEqual(Path(observables.__file__).resolve().parent, LEARN / "analysis")
        with tempfile.TemporaryDirectory() as scratch:
            run = Path(scratch)
            (run / "settings.json").write_text(json.dumps(dict(
                network="attention", eps=5.0, electrons=18, gradient="full", seed=0, iterations=150000,
                settings=dataclasses.asdict(twisted(table2(tm), THIRD, THIRD)))))
            saved, settings = evaluate_weights.run_settings(run)
            self.assertEqual((settings.twist_1, settings.twist_2), (THIRD, THIRD))
            self.assertEqual(settings, twisted(table2(tm), THIRD, THIRD))


class ShellsTest(absltest.TestCase):

    def test_constants_are_the_trainers(self):
        V, _, phi = tm.moire_constants(5.0)
        np.testing.assert_allclose(twist_shells.V, V, rtol=1e-9)
        np.testing.assert_allclose(twist_shells.PHI, phi, rtol=1e-12)
        np.testing.assert_allclose(twist_shells.MOIRE_G, np.asarray(moire_reciprocal_vectors()), atol=1e-9)
        np.testing.assert_allclose(twist_shells.SUPERCELL, np.asarray(slaternet.supercell_vectors(27)), atol=1e-9)
        np.testing.assert_allclose(twist_shells.SUPERCELL_G,
                                   slaternet.reciprocal_vectors(np.asarray(slaternet.supercell_vectors(27))), atol=1e-9)

    def test_levels_equal_learns_shell_structure(self):
        shell_structure = load("shell_structure_original",
                               LEARN / "analysis" / "18_electrons_27_sites" / "shell_structure.py")
        for k in (np.zeros(2), np.array([0.3, -1.1]), twist_shells.twist_vector(THIRD, THIRD)):
            np.testing.assert_allclose(twist_shells.band_energies(k)[0], shell_structure.lowest_band(k), rtol=1e-10)

    def test_periodic_is_open_and_one_third_is_closed(self):
        unit = twist_shells.ENERGY_UNIT_MEV
        momenta, energies, second_band = twist_shells.levels(0.0, 0.0)
        self.assertEqual([count for _, count, _ in twist_shells.shells(energies)], [1, 6, 6, 6, 6, 2])
        np.testing.assert_allclose([e * unit for _, _, e in twist_shells.shells(energies)],
                                   [-11.990, -9.387, -4.709, -2.577, 0.167, 2.237], atol=2e-3)
        self.assertLess(twist_shells.gap(energies), 1e-9)                       # 5 electrons in 6 equal levels
        momenta, energies, second_band = twist_shells.levels(THIRD, THIRD)
        self.assertEqual([count for _, count, _ in twist_shells.shells(energies)], [3, 3, 6, 6, 3, 6])
        np.testing.assert_allclose(twist_shells.gap(energies) * unit, 1.1260, atol=1e-4)
        np.testing.assert_allclose(energies[:18].sum() * unit / 18, -6.0338, atol=1e-4)
        self.assertGreater(second_band, energies[-1])
        self.assertTrue(twist_shells.is_threefold_symmetric(momenta))
        self.assertFalse(twist_shells.is_inversion_symmetric(momenta))          # complex wave function needed
        np.testing.assert_allclose(twist_shells.total_momentum(momenta), 0.0, atol=1e-8)
        # The matrix size does not matter at this accuracy.
        np.testing.assert_allclose(twist_shells.levels(THIRD, THIRD, cutoff=8)[1], energies, atol=1e-8)


if __name__ == "__main__":
    absltest.main()
