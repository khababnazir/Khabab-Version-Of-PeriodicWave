"""
Stage 8 check: our networks with the released settings ARE the authors' networks.
Run:  python test_networks.py      (CPU, double precision, a few minutes)

Our Stage 8 trains with our own code (Stages 3-7) but must be comparable with
the reference run made with the authors' code (learn/archive/stage08_authors_code).
So slaternet.init_released / log_psi_released and psiformer.init_released /
log_psi_released must be the networks that run used.  We compare them with
the authors' networks, built from the same settings, for SlaterNet and for
the self-attention network:

  settings      our defaults are the ones the reference runs were trained with
                (read from their config.json);
  parameters    the same arrays with the same shapes: 34,880 and 103,232 numbers;
  initial       the same initial distribution (same scales, layer norm 1 and 0);
  function      with the reference run's TRAINED weights, at its own walkers:
                the same log|Psi| and the same phase (modulo 2 pi);
  derivatives   the same derivatives with respect to the parameters (what the
                gradient and KFAC use) ...
  energy        ... and the same local energy, which checks the first and
                second derivatives with respect to the positions;
  kfac          kfac-jax finds the same layers in both networks, so the
                optimiser treats them the same way.

What this cannot show: a finite set of configurations cannot prove agreement
everywhere, and none of this tests training itself.
"""
import json
import os
import sys
from pathlib import Path

os.environ["JAX_PLATFORMS"] = "cpu"           # before JAX is imported: a GPU may be training
from absl.testing import absltest
import jax
jax.config.update("jax_enable_x64", True)     # before any JAX operation
import numpy as np
import jax.numpy as jnp
import kfac_jax

REPO = Path(__file__).resolve().parents[2]
for stage in ("stage03_slater", "stage04_mcmc", "stage05_attention", "stage07_ewald"):
    sys.path.insert(0, str(REPO / "learn" / stage))
import slaternet  # noqa: E402
import psiformer  # noqa: E402
from local_energy import moire_units, local_energy  # noqa: E402
from ewald import make_ewald_setup, coulomb_energy, moire_interaction_strength  # noqa: E402

# The authors' code, only as the reference to compare with.
from periodicwave import CustomPsiformer, SlaterNet, networks  # noqa: E402
from periodicwave import curvature_tags_and_blocks, envelopes  # noqa: E402
from periodicwave.pbc import feature_layer, hamiltonian, lattices  # noqa: E402

REFERENCE = REPO / "runs" / "archive" / "stage08_authors_code" / "training"
SUPERCELL, MOIRE_LATTICE = lattices._triangular_lattice_vecs_periodic_potential(1.0, 9)
G = slaternet.reciprocal_vectors(SUPERCELL)
N_ELECTRONS = 6
PHI = np.pi / 4
N_WALKERS = 64


# ------------------------------------------------- the authors' parameters -> ours

def ours_from_authors_slaternet(p):
    layers = p["layers"]
    return {
        "embed": {"w": layers["embed"]},
        "layers": [{"perceptrons": [{"w": m["w"], "b": m["b"]} for m in mlp],
                    "norm": dict(ln[0])}
                   for mlp, ln in zip(layers["mlp"], layers["ln"])],
        "orbital": {"w": p["orbital"][0]["w"]},
    }


def ours_from_authors_psiformer(p):
    layers = p["layers"]
    return {
        "embed": {"w": layers["embed"]},
        "layers": [{"attention": {"queries": {"w": attention["q_w"]},
                                  "keys": {"w": attention["k_w"]},
                                  "values": {"w": attention["v_w"]},
                                  "output": {"w": attention["attn_output"]}},
                    "attention_norm": dict(ln[0]),
                    "perceptrons": [{"w": m["w"], "b": m["b"]} for m in mlp],
                    "perceptron_norm": dict(ln[1])}
                   for attention, ln, mlp in zip(layers["attention"], layers["ln"], layers["mlp"])],
        "orbital": {"w": p["orbital"][0]["w"]},
    }


# ------------------------------------------------- the two networks to compare

def authors_network(family, settings, determinants, n_electrons=N_ELECTRONS, supercell=SUPERCELL):
    """The authors' network, built as their train.py builds it for the moire config
    (6 electrons on 9 sites unless told otherwise)."""
    features = feature_layer.make_pbc_feature_layer(
        natoms=1, nspins=(n_electrons, 0), ndim=2, lattice=jnp.asarray(supercell), include_r_ae=False)
    return family.make_fermi_net(
        (n_electrons, 0), jnp.array([0.0]), ndim=2, determinants=determinants,
        envelope=envelopes.make_null_envelope(), feature_layer=features,
        jastrow="NONE", jastrow_kwargs={"ndim": 2}, bias_orbitals=False,
        complex_output=True, pbc_lattice=jnp.asarray(supercell), **settings)


def setup_case(run, family, arch, init_ours, log_psi_ours, to_ours):
    """Everything one comparison needs, from the reference run's folder."""
    config = json.loads((REFERENCE / run / "config.json").read_text())
    network = config["network"]
    theirs = authors_network(family, network[network["network_type"]], network["determinants"])
    saved = np.load(REFERENCE / run / "qmcjax_ckpt_199999.npz", allow_pickle=True)
    # [0]: the checkpoint keeps a leading axis for the one GPU it was trained on.
    trained = jax.tree.map(lambda x: jnp.asarray(x[0], dtype=jnp.float64), saved["params"].tolist())
    walkers = jnp.asarray(saved["data"].tolist()["positions"][0, :N_WALKERS], dtype=jnp.float64)
    return dict(config=config, theirs=theirs, arch=arch, init_ours=init_ours, to_ours=to_ours,
                log_psi_ours=lambda params, R: log_psi_ours(params, R, G, arch),
                trained_theirs=trained, trained_ours=to_ours(trained),
                walkers=walkers.reshape(N_WALKERS, N_ELECTRONS, 2))


def authors_log_psi(case, params, R):
    """(log|Psi|, phase) of the authors' network at one configuration R, shape (N, 2)."""
    phase, log_abs = case["theirs"].apply(params, R.reshape(-1), jnp.ones(N_ELECTRONS),
                                          jnp.zeros((1, 2)), jnp.array([0.0]))
    return log_abs, phase


def shapes(tree):
    return jax.tree.map(lambda x: x.shape, tree)


def flat(tree):
    return np.concatenate([np.ravel(x) for x in jax.tree.leaves(tree)])


def kfac_layers(log_abs_psi, params, batch):
    """Sorted (block type, parameter shapes) of every layer kfac-jax registers,
    with the authors' KFAC settings and layer patterns."""
    def loss(params, key, batch):
        values = jax.vmap(lambda R: log_abs_psi(params, R))(batch)
        kfac_jax.register_normal_predictive_distribution(values[:, None])
        return jnp.mean(values), {}
    optimizer = kfac_jax.Optimizer(
        value_and_grad_func=jax.value_and_grad(loss, has_aux=True), l2_reg=0.0,
        value_func_has_aux=True, value_func_has_rng=True, learning_rate_schedule=lambda t: 0.1,
        curvature_ema=0.95, inverse_update_period=1, min_damping=1e-6, num_burnin_steps=0,
        register_only_generic=False, estimation_mode="fisher_exact", multi_device=False,
        auto_register_kwargs=dict(graph_patterns=curvature_tags_and_blocks.GRAPH_PATTERNS))
    optimizer.init(params, jax.random.PRNGKey(0), batch)
    return sorted((type(block).__name__, tuple(tuple(s) for s in block.parameters_shapes))
                  for block in optimizer.estimator.blocks)


class SameNetworksTest(absltest.TestCase):

    @classmethod
    def setUpClass(cls):
        if not REFERENCE.exists():
            raise FileNotFoundError(f"the reference runs are needed: {REFERENCE}")
        # Tolerances.  Wave functions and their parameter derivatives must agree
        # to rounding for both networks.  (The attention network calls
        # jax.nn.dot_product_attention, as the authors do; a written-out
        # softmax agreed only to about 3e-6, because that function computes
        # its softmax in single precision even for float64 inputs.)
        # The local energy needs second derivatives with respect to the
        # positions; the two codes take them along different routes through
        # that single-precision softmax, so for the attention network they can
        # agree only at single-precision level: measured up to 4.3e-6 (median
        # 1.6e-7) over 64 walkers, allowed 1e-4.  The function and its
        # parameter derivatives are checked at 1e-10 by the other tests; this
        # one is for the Laplacian, moire and Coulomb terms, where a mistake
        # is far larger.
        cls.cases = {
            "SlaterNet": dict(setup_case(
                "slaternet_eps5", SlaterNet,
                slaternet.ReleasedArchitecture(n_electrons=N_ELECTRONS),
                slaternet.init_released, slaternet.log_psi_released, ours_from_authors_slaternet),
                n_parameters=34880, rtol=1e-10, rtol_energy=1e-10),
            "attention": dict(setup_case(
                "psiformer_eps5", CustomPsiformer,
                psiformer.ReleasedAttentionArchitecture(n_electrons=N_ELECTRONS),
                psiformer.init_released, psiformer.log_psi_released, ours_from_authors_psiformer),
                n_parameters=103232, rtol=1e-10, rtol_energy=1e-4),
        }

    def test_settings_are_those_of_the_reference_runs(self):
        for name, case in self.cases.items():
            network, arch = case["config"]["network"], case["arch"]
            settings = network[network["network_type"]]
            with self.subTest(name):
                self.assertEqual(arch.n_layers, settings["num_layers"])
                self.assertEqual(arch.width, settings["mlp_dim"])
                self.assertEqual(arch.perceptrons_per_layer, settings["num_perceptrons_per_layer"])
                self.assertEqual(arch.n_determinants, network["determinants"])
                if name == "attention":
                    self.assertEqual(arch.n_heads, settings["num_heads"])
                    self.assertEqual(arch.attention_dim, settings["attn_dim"])
                    self.assertEqual(arch.value_dim, settings["value_dim"])
                # Built into our code rather than set by a number:
                self.assertTrue(settings["use_layer_norm"])
                self.assertEqual(settings["mlp_activation_fct"], "GELU")
                self.assertTrue(network["complex"])
                self.assertEqual(network["jastrow"], "NONE")
                self.assertFalse(network["bias_orbitals"])
                self.assertEqual(network["make_envelope_fn"], "periodicwave.envelopes.make_null_envelope")
                self.assertFalse(network["make_feature_layer_kwargs"]["include_r_ae"])

    def test_same_parameters(self):
        for name, case in self.cases.items():
            ours = case["init_ours"](jax.random.key(0), case["arch"])
            theirs = case["to_ours"](case["theirs"].init(jax.random.PRNGKey(0)))
            with self.subTest(name):
                self.assertEqual(shapes(ours), shapes(theirs))
                self.assertEqual(flat(ours).size, case["n_parameters"])

    def test_same_initial_distribution(self):
        """Mean square of every parameter array, averaged over 64 initialisations.

        Both draw weights from normal / sqrt(inputs) and biases from normal.
        Sampling noise: the smallest random arrays are 64 biases.  The mean
        square of 64 normal numbers fluctuates by sqrt(2/64) = 18%; averaged
        over 64 initialisations, 2.2%; the difference of ours and theirs, 3%.
        15% is five times that.  A wrong scale is off by far more: our Stage 3
        choice of 0.1 would give a mean square 100 times too small.
        (A first version averaged over only 8 initialisations, noise 9%, with
        a 20% tolerance, and failed once by chance at 28%.)
        Layer norms start at exactly scale 1 and offset 0.
        """
        for name, case in self.cases.items():
            def mean_square(init):
                samples = [jax.tree.map(lambda x: float(jnp.mean(x ** 2)), init(seed)) for seed in range(64)]
                return np.mean([jax.tree.leaves(s) for s in samples], axis=0)
            ours = mean_square(lambda s: case["init_ours"](jax.random.key(s), case["arch"]))
            theirs = mean_square(lambda s: case["to_ours"](case["theirs"].init(jax.random.PRNGKey(s))))
            with self.subTest(name):
                np.testing.assert_allclose(ours, theirs, rtol=0.15)

    def test_trained_weights_give_the_same_wave_function(self):
        for name, case in self.cases.items():
            ours = jax.vmap(lambda R: case["log_psi_ours"](case["trained_ours"], R))(case["walkers"])
            log_abs, phase = jax.vmap(lambda R: authors_log_psi(case, case["trained_theirs"], R))(case["walkers"])
            with self.subTest(name):
                np.testing.assert_allclose(ours.log_abs, log_abs, rtol=case["rtol"])
                # Phases are compared modulo 2 pi.
                np.testing.assert_allclose(np.exp(1j * (ours.phase - phase)), 1.0, atol=case["rtol"])

    def test_same_derivatives_with_respect_to_the_parameters(self):
        for name, case in self.cases.items():
            R = case["walkers"][0]
            for part, label in ((0, "log|Psi|"), (1, "phase")):
                ours = jax.grad(lambda p: case["log_psi_ours"](p, R)[part])(case["trained_ours"])
                theirs = jax.grad(lambda p: authors_log_psi(case, p, R)[part])(case["trained_theirs"])
                a, b = flat(ours), flat(case["to_ours"](theirs))
                with self.subTest(network=name, of=label):
                    np.testing.assert_allclose(a, b, rtol=case["rtol"], atol=case["rtol"] * np.abs(b).max())

    def test_same_local_energy(self):
        """Ours: Stage 4 kinetic + moire and Stage 7 Coulomb, on our network.
        Theirs: their local energy with the 'default' Laplacian (complex), on theirs."""
        _, V = moire_units()
        U = moire_interaction_strength(dielectric_constant=5.0)
        ewald = make_ewald_setup(SUPERCELL)
        for name, case in self.cases.items():
            authors_e_loc = hamiltonian.local_energy(
                case["theirs"].apply, jnp.array([0.0]), (N_ELECTRONS, 0), use_scan=False,
                complex_output=True, lattice=jnp.asarray(SUPERCELL), potential_type="CoulombMoire",
                potential_kwargs={"moire_lattice_vectors": MOIRE_LATTICE, "moire_potential_strength": V,
                                  "moire_potential_phi": PHI, "interaction_energy_scale": U},
                kinetic_kwargs={"laplacian_method": "default"})
            psi = lambda R: case["log_psi_ours"](case["trained_ours"], R)              # noqa: E731
            ours, theirs = [], []
            for R in case["walkers"][:4]:
                ours.append(complex(local_energy(psi, R, V, PHI) + coulomb_energy(ewald, R, strength=U)))
                data = networks.FermiNetData(positions=R.reshape(-1), spins=jnp.ones(N_ELECTRONS),
                                             atoms=jnp.zeros((1, 2)), charges=jnp.array([0.0]))
                theirs.append(complex(authors_e_loc(case["trained_theirs"], None, data)[0]))
            ours, theirs = np.array(ours), np.array(theirs)
            with self.subTest(name):
                tolerance = case["rtol_energy"]
                np.testing.assert_allclose(ours.real, theirs.real, rtol=tolerance)
                np.testing.assert_allclose(ours.imag, theirs.imag, atol=tolerance * np.abs(theirs.real).max())

    def test_kfac_finds_the_same_layers(self):
        """A layer kfac-jax did not recognise would get its crude fallback
        block (NaiveDiagonal), with no error message.  So both networks must
        give the same layer list, every parameter must be in it, and only the
        two expected kinds of block may appear: dense layers applied to every
        electron, and the scale and offset of the layer norms."""
        for name, case in self.cases.items():
            batch = case["walkers"][:8]
            ours = kfac_layers(lambda p, R: case["log_psi_ours"](p, R).log_abs, case["trained_ours"], batch)
            theirs = kfac_layers(lambda p, R: authors_log_psi(case, p, R)[0], case["trained_theirs"], batch)
            with self.subTest(name):
                self.assertEqual(ours, theirs)
                self.assertEqual({block for block, _ in ours}, {"RepeatedDenseBlock", "ScaleAndShiftDiagonal"})
                covered = sum(int(np.prod(s)) for _, block_shapes in ours for s in block_shapes)
                self.assertEqual(covered, case["n_parameters"])


if __name__ == "__main__":
    absltest.main()
