"""Time one training iteration of train_moire.py's pieces at N electrons on S sites.  Writes nothing."""
import json, sys, time
sys.path.insert(0, "/home/khabab/moire-nnvmc/learn/stage08_own_code")
import train_moire as tm
network, n_el, n_sites, batch = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
sizes = json.loads(sys.argv[5]) if len(sys.argv) > 5 else {}
lock = tm.hold_lock(tm.GPU_LOCKS / "gpu0.lock", "GPU 0 is in use")      # noqa: F841
import jax
import jax.numpy as jnp
assert jax.default_backend() == "gpu"
jax.config.update("jax_default_matmul_precision", "float32")
import slaternet, psiformer
from local_energy import local_energy
from ewald import make_ewald_setup, coulomb_energy

lattice = slaternet.supercell_vectors(n_sites)
G = slaternet.reciprocal_vectors(lattice)
if network == "slaternet":
    arch, init, lp = slaternet.ReleasedArchitecture(n_electrons=n_el, **sizes), slaternet.init_released, slaternet.log_psi_released
else:
    arch, init, lp = psiformer.ReleasedAttentionArchitecture(n_electrons=n_el, **sizes), psiformer.init_released, psiformer.log_psi_released
V, U, phi = tm.moire_constants(5.0)
ewald = make_ewald_setup(lattice)
log_psi = lambda params, R: lp(params, R, G, arch)                                   # noqa: E731
e_loc = lambda params, R: (local_energy(lambda X: log_psi(params, X), R, V, phi)       # noqa: E731
                           + coulomb_energy(ewald, R, strength=U))
problem = tm.Problem(arch, init, log_psi, e_loc)
settings = tm.Settings(batch=batch)
_, sample_and_measure = tm.make_sampler(problem, settings, "real")
optimizer = tm.make_optimizer(tm.make_loss(problem, "real"), settings)
key, k_params, k_walkers = jax.random.split(jax.random.PRNGKey(0), 3)
params = problem.init(k_params, problem.arch)
walkers = jax.random.uniform(k_walkers, (batch, n_el, 2)) @ jnp.asarray(lattice).T    # spread over the cell
opt_state = optimizer.init(params, key, (walkers, jnp.zeros(batch)))
times = []
for i in range(6):
    start = time.time()
    key, k1, k2 = jax.random.split(key, 3)
    walkers, e, stats = sample_and_measure(params, walkers, k1, 0.1)
    params, opt_state, _ = optimizer.step(params, opt_state, k2, batch=(walkers, e),
                                          momentum=jnp.asarray(0.0), damping=jnp.asarray(1e-3))
    jax.block_until_ready(params)
    times.append(time.time() - start)
peak = jax.devices()[0].memory_stats().get("peak_bytes_in_use", 0) / 2 ** 30
print(json.dumps(dict(network=network, electrons=n_el, sites=n_sites, batch=batch, sizes=sizes,
                      parameters=int(sum(x.size for x in jax.tree.leaves(params))),
                      first_iteration_s=round(times[0], 1), seconds_per_iteration=round(sum(times[1:]) / 5, 3),
                      peak_GiB=round(peak, 2))), flush=True)
