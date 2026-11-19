"""
Checks before the eps = 3 seed-42 runs.  Run, from this folder (CPU, about a minute):
    ../../../.venv/bin/python test_honeycomb_field.py

  field       on the 27 sites: 3 lambda on sublattice 0 (as learn's checkpoint_structure labels it),
              -1.5 lambda on sublattices 1 and 2; periodic in the supercell; a full honeycomb with
              sublattice 0 empty gains 27 lambda;
  energy      built exactly as their train.py builds it, from each script's configuration: the
              field script's local energy minus the plain seed-42 script's equals the field term,
              for random weights and positions; with lambda = 0 the two are identical;
  configs     the seed-42 script's configuration differs from the configuration the first eps = 3
              self-attention run's code wrote (its config.json) only in debug.deterministic; the
              field script's differs from the seed-42 script's only in the Hamiltonian's name, the
              field strength and the number of iterations;
  start       both scripts give the same initial weights (their network.init from seed 42).
"""
import json
import os
import sys
from pathlib import Path

os.environ["JAX_PLATFORMS"] = "cpu"
from absl.testing import absltest
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))                                            # honeycomb_field.py
sys.path.insert(0, str(REPO / "authors_exact_code_runs" / "code"))       # evaluate_authors_weights.py
sys.path.insert(0, str(REPO / "learn" / "stage08_own_code" / "analysis" / "18_electrons_27_sites"))
import honeycomb_field as hf                  # noqa: E402
import evaluate_authors_weights as ev         # noqa: E402
import checkpoint_structure as cs             # noqa: E402

ARGUMENTS = ["18_0", "27", "0.35", "0.3333333333333333", "8.031", "15", "45", "CustomPsiformer"]
SEED = HERE / "run_2deg_triangular_potential_seed42.py"
FIELD = HERE / "run_2deg_triangular_potential_seed42_field.py"
FIRST_RUN = (REPO / "authors_exact_code_runs" / "runs" / "18_electrons_27_sites_eps3_released_settings" / "results"
             / "2deg-CoulombMoire" / "CustomPsiformer" / "el18_0_N27_V4.44370908_45.0_U17.70578894" / "config.json")


def flat(tree, prefix=""):
    out = {}
    for key, value in tree.items():
        name = f"{prefix}.{key}" if prefix else key
        out.update(flat(value, name) if isinstance(value, dict) else {name: value})
    return out


def as_json(cfg):
    import tempfile
    from periodicwave.utils import custom_logging
    with tempfile.TemporaryDirectory() as scratch:
        custom_logging.save_config_dict_as_json(cfg, os.path.join(scratch, "config.json"))
        return json.loads(Path(scratch, "config.json").read_text())


class FieldTest(absltest.TestCase):

    def test_values_on_the_sites(self):
        supercell, sites, kappas, sublattice = cs.geometry()
        for strength in (0.5, 1.3):
            per_site = np.array([float(hf.field_energy(np.asarray(s), strength)) for s in sites])
            np.testing.assert_allclose(per_site[sublattice == 0], 3 * strength, atol=1e-9)
            np.testing.assert_allclose(per_site[sublattice != 0], -1.5 * strength, atol=1e-9)
            self.assertEqual(int(np.sum(sublattice == 0)), 9)
            honeycomb = sites[sublattice != 0]                                # 18 electrons, sublattice 0 empty
            np.testing.assert_allclose(float(hf.field_energy(honeycomb, strength)), -27 * strength, atol=1e-8)
            even = np.concatenate([sites[sublattice == g][:6] for g in range(3)])   # 6 per sublattice
            np.testing.assert_allclose(float(hf.field_energy(even, strength)), 0.0, atol=1e-8)

    def test_periodic_in_the_supercell(self):
        supercell = np.asarray(cs.geometry()[0])
        r = np.random.default_rng(0).uniform(-3, 3, size=(18, 2))
        for i in (0, 1):
            np.testing.assert_allclose(float(hf.field_energy(r + supercell[:, i], 0.5)),
                                       float(hf.field_energy(r, 0.5)), atol=1e-9)


class EnergyTest(absltest.TestCase):

    def test_field_script_adds_exactly_the_field(self):
        import jax
        import jax.numpy as jnp
        from periodicwave import networks
        cfg_seed = ev.authors_configuration(SEED, ARGUMENTS)
        cfg_field = ev.authors_configuration(FIELD, ARGUMENTS)
        self.assertEqual(cfg_field.system.make_local_energy_fn, "honeycomb_field.local_energy")
        strength = cfg_field.system.make_local_energy_kwargs["honeycomb_field_strength"]
        self.assertEqual(strength, 0.5)
        net_seed, e_seed, charges = ev.build_like_train_py(cfg_seed)
        net_field, e_field, _ = ev.build_like_train_py(cfg_field)
        params = net_seed.init(jax.random.PRNGKey(3))
        supercell = np.asarray(cfg_seed.system.make_local_energy_kwargs["lattice"])
        positions = jax.random.uniform(jax.random.PRNGKey(4), (3, 18, 2)) @ jnp.asarray(supercell).T
        for x in positions:
            data = networks.FermiNetData(positions=x.reshape(-1), spins=jnp.ones(18), atoms=jnp.zeros((1, 2)),
                                         charges=charges)
            a = complex(e_seed(params, None, data)[0])
            b = complex(e_field(params, None, data)[0])
            np.testing.assert_allclose(b - a, float(hf.field_energy(x, strength)), rtol=0, atol=1e-3)
            self.assertGreater(abs(float(hf.field_energy(x, strength))), 1e-3)
        cfg_zero = ev.authors_configuration(FIELD, ARGUMENTS)
        cfg_zero.system.make_local_energy_kwargs["honeycomb_field_strength"] = 0.0
        _, e_zero, _ = ev.build_like_train_py(cfg_zero)
        data = networks.FermiNetData(positions=positions[0].reshape(-1), spins=jnp.ones(18), atoms=jnp.zeros((1, 2)),
                                     charges=charges)
        self.assertEqual(complex(e_zero(params, None, data)[0]), complex(e_seed(params, None, data)[0]))


class ConfigTest(absltest.TestCase):

    def test_seed_script_differs_from_the_first_run_only_in_the_seed(self):
        rebuilt = flat(as_json(ev.authors_configuration(SEED, ARGUMENTS)))
        first = flat(json.loads(FIRST_RUN.read_text()))
        differing = {k: (first.get(k), rebuilt.get(k)) for k in set(first) | set(rebuilt) if first.get(k) != rebuilt.get(k)}
        self.assertEqual(differing, {"debug.deterministic": (False, True)})

    def test_field_script_differs_only_in_the_field(self):
        seed = flat(as_json(ev.authors_configuration(SEED, ARGUMENTS)))
        field = flat(as_json(ev.authors_configuration(FIELD, ARGUMENTS)))
        differing = {k for k in set(seed) | set(field) if seed.get(k) != field.get(k)}
        self.assertEqual(differing, {"system.make_local_energy_fn", "optim.iterations",
                                     "system.make_local_energy_kwargs.honeycomb_field_strength"})
        self.assertEqual(field["optim.iterations"], 20000)
        self.assertEqual(seed["optim.iterations"], 200000)


class StartTest(absltest.TestCase):

    def test_same_initial_weights(self):
        """Their train.py: key = PRNGKey(42), split once, network.init(subkey) (train.py lines 368-437)."""
        import jax
        from periodicwave import train
        trees = []
        for script in (SEED, FIELD):
            cfg = ev.authors_configuration(script, ARGUMENTS)
            self.assertTrue(cfg.debug.deterministic)
            network, _, _ = ev.build_like_train_py(cfg)
            _, subkey = jax.random.split(jax.random.PRNGKey(42))
            trees.append(jax.tree_util.tree_leaves(network.init(subkey)))
        for a, b in zip(*trees):
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


if __name__ == "__main__":
    absltest.main()
