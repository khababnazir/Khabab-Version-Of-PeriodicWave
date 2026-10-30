"""Time and peak memory of ONE piece of a training iteration, 18 electrons, 27 sites, eps 5 (smoke test, /tmp only)."""
import json, sys, time
side, piece, batch = sys.argv[1], sys.argv[2], int(sys.argv[3])
sys.path.insert(0, "/home/khabab/moire-nnvmc/learn/stage08_own_code"); sys.path.insert(0, "/tmp")
import train_moire as tm
lock = tm.hold_lock(tm.GPU_LOCKS / "gpu0.lock", "GPU 0 is in use")      # noqa: F841
import jax
import jax.numpy as jnp
assert jax.default_backend() == "gpu"
jax.config.update("jax_default_matmul_precision", "float32")
import slaternet
lattice = slaternet.supercell_vectors(27)

import os
if os.environ.get("DETERMINANT") == "slogdet":
    # Test only: add the determinants from slogdet, as the authors' logdet_matmul does,
    # instead of Stage 3's plain jnp.linalg.det.
    import slaternet as _s, psiformer as _p
    def sum_of_determinants_slogdet(matrices):
        sign, logdet = jnp.linalg.slogdet(matrices)
        shift = jax.lax.stop_gradient(jnp.max(logdet))
        total = jnp.sum(sign * jnp.exp(logdet - shift))
        return _s.LogPsi(log_abs=shift + jnp.log(jnp.abs(total)), phase=jnp.angle(total))
    _s.sum_of_determinants = sum_of_determinants_slogdet
    _p.sum_of_determinants = sum_of_determinants_slogdet
walkers = jax.random.uniform(jax.random.PRNGKey(3), (batch, 18, 2)) @ jnp.asarray(lattice).T

if side == "ours":
    import psiformer
    from local_energy import local_energy, moire_potential
    from ewald import make_ewald_setup, coulomb_energy
    from folx_kinetic import kinetic_energy_folx
    G = slaternet.reciprocal_vectors(lattice)
    arch = psiformer.ReleasedAttentionArchitecture(n_electrons=18)
    log_psi = lambda params, R: psiformer.log_psi_released(params, R, G, arch)                   # noqa: E731
    V, U, phi = tm.moire_constants(5.0)
    ewald = make_ewald_setup(lattice)
    params = psiformer.init_released(jax.random.PRNGKey(1), arch)
    if piece == "logpsi":
        fn = jax.jit(lambda p, W: jax.vmap(lambda R: log_psi(p, R).log_abs)(W))
    elif piece == "eloc_hessian":
        fn = jax.jit(lambda p, W: jax.vmap(lambda R: local_energy(lambda X: log_psi(p, X), R, V, phi)
                                           + coulomb_energy(ewald, R, strength=U))(W))
    elif piece == "eloc_folx":
        fn = jax.jit(lambda p, W: jax.vmap(lambda R: kinetic_energy_folx(lambda X: log_psi(p, X), R)
                                           + moire_potential(R, V, phi) + coulomb_energy(ewald, R, strength=U))(W))
    elif piece in ("move", "kfac"):
        problem = tm.Problem(arch, psiformer.init_released, log_psi, None)
        settings = tm.Settings(batch=batch)
        move, _ = tm.make_sampler(problem, settings, "real")
        if piece == "move":
            fn = lambda p, W: move(p, W, jax.random.PRNGKey(4), 0.1)                            # noqa: E731
        else:
            optimizer = tm.make_optimizer(tm.make_loss(problem, "real"), settings)
            energies = jax.random.normal(jax.random.PRNGKey(5), (batch,))
            state = {"opt": optimizer.init(params, jax.random.PRNGKey(6), (walkers, energies))}
            def fn(p, W):
                new_p, state["opt"], _ = optimizer.step(p, state["opt"], jax.random.PRNGKey(7), batch=(W, energies),
                                                        momentum=jnp.asarray(0.0), damping=jnp.asarray(1e-3))
                return new_p
else:   # the authors' network and local energy, built as their train.py builds them
    sys.path.insert(0, "/home/khabab/moire-nnvmc/learn/archive/stage08_authors_code")
    sys.path.insert(0, "/home/khabab/moire-nnvmc/learn/archive/stage08_authors_code/analysis")
    import run_moire
    from evaluate_weights import build_like_train_py
    from periodicwave import networks
    cfg = run_moire.authors_configuration(["18_0", "27", "0.35", repr(1.0 / 5.0), "8.031", "15", "45", "CustomPsiformer"])
    network, local_energy_fn, charges = build_like_train_py(cfg)
    params = network.init(jax.random.PRNGKey(1))
    data = networks.FermiNetData(positions=walkers.reshape(batch, -1), spins=jnp.ones((batch, 18)),
                                 atoms=jnp.zeros((batch, 1, 2)), charges=jnp.zeros((batch, 1)))
    axes = networks.FermiNetData(positions=0, spins=0, atoms=0, charges=0)
    if piece == "logpsi":
        fn = jax.jit(lambda p, d: jax.vmap(lambda x, s, a, c: network.apply(p, x, s, a, c)[1])(d.positions, d.spins, d.atoms, d.charges))
    elif piece == "eloc_folx":
        fn = jax.jit(jax.vmap(lambda p, d: local_energy_fn(p, None, d)[0], in_axes=(None, axes)))
    walkers = data

n_params = int(sum(x.size for x in jax.tree.leaves(params)))
times = []
for i in range(6):
    start = time.time()
    out = fn(params, walkers)
    if piece == "kfac":
        params = out
    jax.block_until_ready(out)
    times.append(time.time() - start)
peak = jax.devices()[0].memory_stats().get("peak_bytes_in_use", 0) / 2 ** 30
print(json.dumps(dict(determinant=os.environ.get("DETERMINANT", "det"), side=side, piece=piece, batch=batch, parameters=n_params, first_call_s=round(times[0], 1),
                      seconds_per_call=round(sum(times[1:]) / 5, 4), peak_GiB=round(peak, 2))), flush=True)
