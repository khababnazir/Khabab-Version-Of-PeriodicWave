"""
Stage 8 checks for train_moire.py.  Run:  python test_train_moire.py   (CPU, double precision, a few minutes)

  settings    every setting equals the reference run's config.json, at eps = 5
              and 10, the moire potential and the Coulomb strength included;
  gradient    what KFAC receives is Stage 6's energy_gradient, in both
              --gradient modes;
  one update  from the reference run's TRAINED weights and 64 of its walkers,
              one complete update of ours equals one of the authors' code:
              the local energies; then, given the same local energies, the
              clipping, the gradient and the new parameters after one KFAC
              step.  This is the whole optimiser, not only its layers;
  restart     stop and restart gives exactly the run that never stopped, and
              a folder holding a different experiment is refused.

What this cannot show: that a long run converges.  Two runs with different
random numbers (initial weights, Metropolis moves) follow different paths;
whether they end at the same energy is what the runs themselves test.
"""
import csv
import dataclasses
import json
import os
import tempfile
from pathlib import Path

os.environ["JAX_PLATFORMS"] = "cpu"           # before JAX is imported: a GPU may be training
from absl.testing import absltest
import jax
jax.config.update("jax_enable_x64", True)     # before any JAX operation
import numpy as np
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import kfac_jax

import train_moire as tm
import test_networks as reference             # the reference run and the authors' networks
from gradient import energy_gradient, log_derivatives, clip_local_energy  # Stage 6

from periodicwave import constants, curvature_tags_and_blocks, networks  # noqa: E402
from periodicwave import loss as authors_loss  # noqa: E402
from periodicwave.pbc import hamiltonian  # noqa: E402

N_WALKERS = 64


def flat(tree):
    return np.asarray(ravel_pytree(tree)[0])


def config(run):
    return json.loads((reference.REFERENCE / run / "config.json").read_text())


class SettingsTest(absltest.TestCase):

    def test_settings_are_those_of_the_reference_runs(self):
        s = tm.Settings()
        for eps, run in ((5.0, "slaternet_eps5"), (10.0, "slaternet_eps10")):
            cfg = config(run)
            optim, mcmc, kfac = cfg["optim"], cfg["mcmc"], cfg["optim"]["kfac"]
            with self.subTest(run):
                self.assertEqual(s.batch, cfg["batch_size"])
                self.assertEqual(s.mcmc_steps, mcmc["steps"])
                self.assertEqual(s.burn_in, mcmc["burn_in"])
                self.assertEqual(s.init_width, mcmc["init_width"])
                self.assertEqual(s.move_width, mcmc["move_width"])
                self.assertEqual(s.adapt_every, mcmc["adapt_frequency"])
                self.assertEqual((mcmc["move_width_updater"], mcmc["blocks"]), ("adaptive", 1))
                self.assertEqual(optim["iterations"], 200000)
                self.assertEqual(optim["optimizer"], "kfac")
                self.assertEqual(s.learning_rate, optim["lr"]["rate"])
                self.assertEqual(s.learning_rate_delay, optim["lr"]["delay"])
                self.assertEqual(s.learning_rate_decay, optim["lr"]["decay"])
                self.assertEqual(s.damping, kfac["damping"])
                self.assertEqual(s.norm_constraint, kfac["norm_constraint"])
                self.assertEqual(s.curvature_ema, kfac["cov_ema_decay"])
                self.assertEqual(s.min_damping, kfac["min_damping"])
                self.assertEqual((kfac["invert_every"], kfac["momentum"], kfac["l2_reg"]), (1, 0.0, 0.0))
                self.assertFalse(kfac["register_only_generic"])
                self.assertEqual(s.clip_scale, optim["clip_local_energy"])
                self.assertEqual((optim["clip_median"], optim["center_at_clip"]), (False, True))
                # The physics: moire potential and Coulomb strength.
                V, U, phi = tm.moire_constants(eps)
                potential = cfg["system"]["make_local_energy_kwargs"]["potential_kwargs"]
                np.testing.assert_allclose(V, potential["moire_potential_strength"], rtol=1e-12)
                np.testing.assert_allclose(U, potential["interaction_energy_scale"], rtol=1e-12)
                np.testing.assert_allclose(phi, potential["moire_potential_phi"], rtol=1e-12)
                self.assertEqual(cfg["system"]["electrons"], [6, 0])


class GradientTest(absltest.TestCase):

    def test_kfac_receives_the_stage6_gradient(self):
        problem = tm.make_problem("slaternet", 5.0)
        params = problem.init(jax.random.PRNGKey(0), problem.arch)
        walkers = jax.random.uniform(jax.random.PRNGKey(1), (8, 6, 2)) @ jnp.asarray(
            reference.SUPERCELL).T
        e_loc = jax.vmap(lambda R: problem.local_energy(params, R))(walkers)
        O = log_derivatives(problem.log_psi, params, walkers)                    # Stage 6
        for gradient in ("real", "full"):
            if gradient == "real":
                e_clipped = jnp.real(clip_local_energy(jnp.real(e_loc)))
            else:
                e_clipped = clip_local_energy(e_loc)
            loss = tm.make_loss(problem, gradient)
            ours = jax.grad(lambda p: loss(p, None, (walkers, e_clipped))[0])(params)
            stage6 = energy_gradient(e_clipped, O)
            with self.subTest(gradient):
                np.testing.assert_allclose(flat(ours), stage6, rtol=1e-10, atol=1e-12 * np.abs(stage6).max())


def authors_update(case, walkers, cfg):
    """Local energies, gradient and one KFAC step of the authors' code, as their train.py does it."""
    theirs = case["theirs"]
    potential = cfg["system"]["make_local_energy_kwargs"]["potential_kwargs"]
    e_l = hamiltonian.local_energy(
        theirs.apply, jnp.array([0.0]), (6, 0), use_scan=False, complex_output=True,
        lattice=jnp.asarray(reference.SUPERCELL), potential_type="CoulombMoire",
        potential_kwargs=dict(potential, moire_lattice_vectors=reference.MOIRE_LATTICE),
        kinetic_kwargs={"laplacian_method": "folx"})

    def log_network(*args):                                   # train.py's log_network
        phase, mag = theirs.apply(*args)
        return mag + 1.0j * phase
    optim = cfg["optim"]
    total_energy = authors_loss.make_loss(
        log_network, e_l, clip_local_energy=optim["clip_local_energy"],
        clip_from_median=optim["clip_median"], center_at_clipped_energy=optim["center_at_clip"],
        complex_output=True)

    n = walkers.shape[0]
    data = networks.FermiNetData(positions=walkers.reshape(n, -1), spins=jnp.ones((n, 6)),
                                 atoms=jnp.zeros((n, 1, 2)), charges=jnp.zeros((n, 1)))
    walker_axes = networks.FermiNetData(positions=0, spins=0, atoms=0, charges=0)
    energies = jax.vmap(lambda d: e_l(case["trained_theirs"], None, d)[0], in_axes=(walker_axes,))(data)

    # One device, as the reference run: their code needs the device axis (pmap).
    replicate = kfac_jax.utils.replicate_all_local_devices
    params, batch = replicate(case["trained_theirs"]), replicate(data)
    key = kfac_jax.utils.make_different_rng_key_on_all_devices(jax.random.PRNGKey(0))
    gradient = constants.pmap(jax.grad(lambda p, k, d: total_energy(p, k, d)[0]))(params, key, batch)

    kfac = optim["kfac"]
    lr = optim["lr"]
    optimizer = kfac_jax.Optimizer(                           # as train.py lines 568-587
        jax.value_and_grad(total_energy, argnums=0, has_aux=True),
        l2_reg=kfac["l2_reg"], norm_constraint=kfac["norm_constraint"],
        value_func_has_aux=True, value_func_has_rng=True,
        learning_rate_schedule=lambda t: lr["rate"] * jnp.power(1.0 / (1.0 + t / lr["delay"]), lr["decay"]),
        curvature_ema=kfac["cov_ema_decay"], inverse_update_period=kfac["invert_every"],
        min_damping=kfac["min_damping"], num_burnin_steps=0,
        register_only_generic=kfac["register_only_generic"], estimation_mode="fisher_exact",
        multi_device=True, pmap_axis_name=constants.PMAP_AXIS_NAME,
        auto_register_kwargs=dict(graph_patterns=curvature_tags_and_blocks.GRAPH_PATTERNS))
    state = optimizer.init(params, key, batch)
    new_params, _, _ = optimizer.step(params=params, state=state, rng=key, batch=batch,
                                      momentum=replicate(jnp.zeros([])),
                                      damping=replicate(jnp.asarray(kfac["damping"])))
    first = lambda tree: jax.tree.map(lambda x: x[0], tree)    # noqa: E731
    return np.asarray(energies), first(gradient), first(new_params)


class OneUpdateTest(absltest.TestCase):

    def test_one_update_equals_the_authors(self):
        """Each piece on the same inputs, so a difference points at one piece:
        our local energies against theirs; then our clipping, gradient and
        KFAC step, given THEIR local energies, against theirs.

        kfac-jax takes over the memory of the parameters it is given (it
        "donates" them), so each step gets its own copy.
        """
        copy = lambda tree: jax.tree.map(jnp.copy, tree)          # noqa: E731
        for name, network, run in (("SlaterNet", "slaternet", "slaternet_eps5"),
                                   ("attention", "attention", "psiformer_eps5")):
            case = reference.SameNetworksTest.cases[name]
            walkers = case["walkers"][:N_WALKERS]
            their_energies, their_gradient, their_params = authors_update(
                dict(case, trained_theirs=copy(case["trained_theirs"])), walkers, config(run))

            problem = tm.make_problem(network, 5.0)
            params = case["trained_ours"]
            before = flat(params)
            settings = tm.Settings()
            e_loc = jax.vmap(lambda R: problem.local_energy(params, R))(walkers)
            e_clipped = jnp.real(clip_local_energy(jnp.asarray(their_energies.real), settings.clip_scale))
            loss = tm.make_loss(problem, "real")
            gradient = jax.grad(lambda p: loss(p, None, (walkers, e_clipped))[0])(params)
            optimizer = tm.make_optimizer(loss, settings)
            key = jax.random.PRNGKey(0)
            state = optimizer.init(params, key, (walkers, e_clipped))
            new_params, _, _ = optimizer.step(copy(params), state, key, batch=(walkers, e_clipped),
                                              momentum=jnp.asarray(0.0), damping=jnp.asarray(settings.damping))

            ours_step = flat(new_params) - before
            theirs_step = flat(case["to_ours"](their_params)) - before
            theirs_gradient = flat(case["to_ours"](their_gradient))
            with self.subTest(network=name, of="local energies"):
                np.testing.assert_allclose(np.real(e_loc), their_energies.real, rtol=case["rtol_energy"])
            with self.subTest(network=name, of="gradient"):
                self.assertLess(relative_difference(flat(gradient), theirs_gradient), UPDATE_TOLERANCE[name])
            with self.subTest(network=name, of="KFAC step"):
                self.assertGreater(np.abs(theirs_step).max(), 0.0)
                self.assertLess(relative_difference(ours_step, theirs_step), UPDATE_TOLERANCE[name])


def relative_difference(a, b):
    """|a - b| / |b| for whole vectors: how far a whole gradient or step is off."""
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


# Measured (64 walkers, double precision): SlaterNet gradient 5e-14 and step
# 5e-13; attention gradient 3e-5 and step 8e-5.  The attention network is
# limited by jax.nn.dot_product_attention's single-precision softmax: the
# authors' loss differentiates forwards (jax.jvp) and ours backwards
# (jax.grad), and in OUR OWN attention network those two already differ by
# up to 1.8e-4 (SlaterNet: 7.5e-14).  A real mistake -- a missing centring,
# a factor 2, the wrong clipping -- changes a gradient by 1e-2 or more.
UPDATE_TOLERANCE = {"SlaterNet": 1e-10, "attention": 1e-3}


class RestartTest(absltest.TestCase):

    def test_restart_continues_exactly(self):
        small = dataclasses.replace(tm.Settings(), batch=16, burn_in=2, mcmc_steps=3)
        quiet = lambda *args: None                              # noqa: E731
        with tempfile.TemporaryDirectory() as scratch:
            straight, restarted = Path(scratch, "straight"), Path(scratch, "restarted")
            straight.mkdir()
            restarted.mkdir()
            tm.train(straight, "slaternet", 10.0, "real", 0, 4, small, log=quiet)
            tm.train(restarted, "slaternet", 10.0, "real", 0, 2, small, log=quiet)    # stops after 2 ...
            tm.train(restarted, "slaternet", 10.0, "real", 0, 4, small, log=quiet)    # ... continues to 4
            with np.load(straight / "checkpoint_000004.npz") as a, np.load(restarted / "checkpoint_000004.npz") as b:
                self.assertEqual(sorted(a.files), sorted(b.files))
                for name in a.files:
                    np.testing.assert_array_equal(a[name], b[name], err_msg=name)

            def rows(folder):
                with open(folder / "train_stats.csv", newline="") as handle:
                    return [{k: v for k, v in row.items() if k != "seconds"} for row in csv.DictReader(handle)]
            self.assertEqual(rows(straight), rows(restarted))
            self.assertEqual([int(r["step"]) for r in rows(restarted)], [0, 1, 2, 3])

    def test_a_different_experiment_is_refused(self):
        wanted = dict(network="slaternet", eps=10.0, gradient="real", seed=0, iterations=4,
                      settings=dataclasses.asdict(tm.Settings()))
        with tempfile.TemporaryDirectory() as scratch:
            folder = Path(scratch)
            self.assertEqual(tm.same_experiment_or_stop(folder, wanted, None), 4)
            self.assertEqual(tm.same_experiment_or_stop(folder, wanted, None), 4)   # the same: fine
            with self.assertRaises(SystemExit):
                tm.same_experiment_or_stop(folder, dict(wanted, eps=5.0), None)
            self.assertEqual(tm.same_experiment_or_stop(folder, dict(wanted, iterations=9), 9), 9)


def setUpModule():
    reference.SameNetworksTest.setUpClass()       # loads the reference runs once


if __name__ == "__main__":
    absltest.main()
