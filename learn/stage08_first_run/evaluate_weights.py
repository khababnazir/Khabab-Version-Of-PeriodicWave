"""
Stage 8 -- evaluate a trained network with its weights FROZEN.

Run (one GPU, one evaluation):
    CUDA_VISIBLE_DEVICES=0 python evaluate_weights.py ../../runs/stage08/psiformer_eps5
Options: --steps N     evaluation steps after burn-in (default 10000)
         --burn-in N   steps thrown away first (default 200)
         --out DIR     default runs/stage08/evaluation/<run name>

Why a separate evaluation.  The energies in train_stats.csv were measured
while the parameters were still changing, one small update per iteration,
so each number belongs to a slightly different wave function.  Here the
parameters are fixed at the final checkpoint and only the walkers move.
That gives the energy of ONE wave function, as a plain Monte Carlo average
with a clean error bar, and on the same samples everything the paper's
Fig. 6 shows (observables.py).

What is the authors' and what is ours.  The wave function, the sampler and
the local energy are the authors' own code, built exactly as their train.py
builds them, from the configuration the run was trained with.  We add only:
  - the move width of the walkers is held FIXED at its final training value
    (training keeps adapting it; with a fixed width the sampler is a
    textbook Metropolis chain whose stationary distribution is |Psi|^2);
  - the energy split into kinetic, moire and Coulomb parts.  The two
    potential terms come from the same functions of pbc/hamiltonian.py
    that the authors' local energy calls; kinetic = total - potentials;
  - the observables of observables.py, measured after every step.
Like the training, everything runs in single precision (float32).

Before the long run, a consistency check: the mean energy of the first 200
evaluation steps must agree with the last 2000 training steps to within
0.05 (code units, ~0.03 meV per electron).  A wrong network, wrong weights
or wrong Hamiltonian would miss by far more, and the run stops.

Output, in the --out folder (rewritten every 1000 steps, so a crash keeps
what was measured):
    energies.csv      one row per step: batch means of E_loc and its parts,
                      the batch variance of E_loc, and the acceptance pmove
    observables.npz   histograms and sums, added up over all steps
    summary.json      settings, checkpoint, timing, the consistency check
"""
import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RUNS = HERE.parents[1] / "runs" / "stage08"

DENSITY_BINS = 90              # 30 bins per moire cell along each supercell vector
RADIAL_BINS = 150
CHECK_STEPS = 200              # evaluation steps used in the consistency check
CHECK_TRAINING_STEPS = 2000    # last training steps it is compared with
CHECK_TOLERANCE = 0.05         # code units of energy (total, all 6 electrons)
SAVE_EVERY = 1000


def build_like_train_py(cfg):
    """The wave function and local energy, built as the authors' train.py builds them.

    Follows periodicwave/train.py, lines 357-441 (network) and 521-534 (local
    energy), PeriodicWave at 6525d2f, leaving out what only matters for
    training on several devices.  Returns
        network        network.apply(params, x, spins, atoms, charges) -> (phase, log|psi|)
        local_energy   local_energy(params, key, data) -> (E_loc, None)
        charges        the dummy nuclear charge the authors carry (no nuclei here)
    """
    import importlib
    import jax.numpy as jnp
    from periodicwave import CustomPsiformer, SlaterNet, envelopes, networks

    charges = jnp.array([0.0])
    nspins = cfg.system.electrons

    if cfg.network.make_feature_layer_fn:
        module, name = cfg.network.make_feature_layer_fn.rsplit(".", maxsplit=1)
        feature_layer = getattr(importlib.import_module(module), name)(
            natoms=charges.shape[0], nspins=nspins, ndim=cfg.system.ndim,
            **cfg.network.make_feature_layer_kwargs)
    else:
        feature_layer = networks.make_ferminet_features(
            natoms=charges.shape[0], nspins=nspins, ndim=cfg.system.ndim)

    if cfg.network.make_envelope_fn:
        module, name = cfg.network.make_envelope_fn.rsplit(".", maxsplit=1)
        envelope = getattr(importlib.import_module(module), name)(**cfg.network.make_envelope_kwargs)
    else:
        envelope = envelopes.make_null_envelope()

    family = {"CustomPsiformer": CustomPsiformer, "SlaterNet": SlaterNet}[cfg.network.network_type]
    network = family.make_fermi_net(
        nspins, charges,
        ndim=cfg.system.ndim,
        determinants=cfg.network.determinants,
        envelope=envelope,
        feature_layer=feature_layer,
        jastrow=cfg.network.get("jastrow", "default"),
        jastrow_kwargs=cfg.network.jastrow_kwargs,
        bias_orbitals=cfg.network.bias_orbitals,
        complex_output=cfg.network.get("complex", False),
        pbc_lattice=cfg.system.pbc_lattice,
        **cfg.network[cfg.network.network_type])

    module, name = cfg.system.make_local_energy_fn.rsplit(".", maxsplit=1)
    local_energy = getattr(importlib.import_module(module), name)(
        f=network.apply, charges=charges, nspins=nspins,
        complex_output=cfg.network.get("complex", False), use_scan=False,
        **cfg.system.make_local_energy_kwargs)
    return network, local_energy, charges


def potential_terms(cfg, charges):
    """Coulomb and moire energy of one configuration, from the authors' own functions.

    The same two calls as in pbc/hamiltonian.local_energy (convergence radius
    5, its default), so total - coulomb - moire is exactly their kinetic energy.
    """
    import jax.numpy as jnp
    from periodicwave import networks
    from periodicwave.pbc import hamiltonian

    settings = cfg.system.make_local_energy_kwargs
    potential = settings["potential_kwargs"]
    coulomb = hamiltonian.make_2DCoulomb_potential(
        settings["lattice"], jnp.array([0.0]), charges,
        settings.get("convergence_radius", 5), potential["interaction_energy_scale"])
    moire = hamiltonian.make_triangular_potential(
        potential["moire_lattice_vectors"], jnp.array([0.0]), charges,
        potential["moire_potential_strength"], potential["moire_potential_phi"])

    def terms(data):
        ae, ee, _, _ = networks.construct_input_features(data.positions, data.atoms, ndim=2)
        return coulomb(ae, ee), moire(ae)
    return terms


def same_structure(a, b):
    """True if two parameter trees have the same layout and array shapes."""
    import jax
    shapes = lambda tree: jax.tree_util.tree_map(lambda x: np.shape(x), tree)     # noqa: E731
    return (jax.tree_util.tree_structure(a) == jax.tree_util.tree_structure(b)
            and jax.tree_util.tree_leaves(shapes(a)) == jax.tree_util.tree_leaves(shapes(b)))


def add_to(totals, sums):
    """Add one step's sums to the running totals, in double precision / 64-bit integers."""
    for name, value in sums.items():
        value = np.asarray(value)
        if np.iscomplexobj(value):
            value = value.astype(np.complex128)
        elif np.issubdtype(value.dtype, np.integer):
            value = value.astype(np.int64)
        else:
            value = value.astype(np.float64)
        totals[name] = value if name not in totals else totals[name] + value


def write_atomically(path, write):
    """Write to path via a temporary file, so a crash never leaves half a file."""
    temporary = path.with_name(path.name + ".tmp")
    write(temporary)
    os.replace(temporary, path)


def save(out, rows, totals, summary):
    def write_csv(path):
        with open(path, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    def write_npz(path):
        with open(path, "wb") as handle:        # a file handle: np.savez would add ".npz" to a name
            np.savez(handle, **totals)
    write_atomically(out / "energies.csv", write_csv)
    write_atomically(out / "observables.npz", write_npz)
    write_atomically(out / "summary.json", lambda p: p.write_text(json.dumps(summary, indent=2)))


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("run", type=Path, help="training folder, e.g. runs/stage08/psiformer_eps5")
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--burn-in", type=int, default=200)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run = args.run.resolve()
    out = (args.out or RUNS / "evaluation" / run.name).resolve()
    out.mkdir(parents=True, exist_ok=True)

    import jax
    import jax.numpy as jnp
    from periodicwave import checkpoint, mcmc, networks
    sys.path.insert(0, str(HERE))
    from run_moire import authors_configuration
    import observables

    # 1. The configuration the run was trained with.
    trained_with = json.loads((run / "overrides.json").read_text())
    cfg = authors_configuration(trained_with["authors_command_line"])
    laplacian = trained_with["overrides"].get(
        "system.make_local_energy_kwargs.kinetic_kwargs.laplacian_method")
    if laplacian is not None:
        cfg.system.make_local_energy_kwargs["kinetic_kwargs"]["laplacian_method"] = laplacian

    # 2. Network and local energy, as train.py builds them; then the weights.
    network, local_energy, charges = build_like_train_py(cfg)
    final = run / f"qmcjax_ckpt_{cfg.optim.iterations - 1:06d}.npz"
    if jax.device_count() != 1:
        sys.exit("make exactly one GPU visible (CUDA_VISIBLE_DEVICES=0): "
                 "the checkpoint was saved from one device")
    _, data, params, _, width = checkpoint.restore(str(final), cfg.batch_size)
    params = jax.tree_util.tree_map(lambda x: jnp.asarray(x)[0], params)      # drop the device axis
    data = jax.tree_util.tree_map(lambda x: jnp.asarray(x)[0], data)
    width = jnp.asarray(width)[0]
    if not same_structure(params, network.init(jax.random.PRNGKey(0))):
        sys.exit(f"{final.name} does not fit the network built from the configuration")

    # 3. One evaluation step: move the walkers, then measure.
    batch = cfg.batch_size
    n_electrons = sum(cfg.system.electrons)
    supercell = np.asarray(cfg.system.make_local_energy_kwargs["lattice"])
    g = 4 * np.pi / np.sqrt(3)                                    # |moire reciprocal vector|, a_M = 1
    qs, q_indices = observables.supercell_q_vectors(supercell, q_max=2 * g)
    r_max = 0.5 * np.linalg.norm(supercell[:, 0])                 # g(r) is valid up to half the cell size

    log_abs_psi = lambda p, x, spins, atoms, charges: network.apply(p, x, spins, atoms, charges)[1]  # noqa: E731
    mcmc_step = mcmc.make_mcmc_step(jax.vmap(log_abs_psi, in_axes=(None, 0, 0, 0, 0)), batch,
                                    steps=cfg.mcmc.steps, ndim=cfg.system.ndim, blocks=cfg.mcmc.blocks)
    walker_axes = networks.FermiNetData(positions=0, spins=0, atoms=0, charges=0)
    batch_local_energy = jax.vmap(lambda p, d: local_energy(p, None, d)[0], in_axes=(None, walker_axes))
    batch_potentials = jax.vmap(potential_terms(cfg, charges), in_axes=(walker_axes,))

    @jax.jit
    def evaluation_step(params, data, key):
        data, pmove = mcmc_step(params, data, key, width)
        energy = jnp.real(batch_local_energy(params, data))
        coulomb, moire = batch_potentials(data)
        means = dict(energy=energy.mean(), variance=energy.var(),
                     kinetic=(energy - coulomb - moire).mean(),
                     moire=moire.mean(), coulomb=coulomb.mean(), pmove=pmove)
        R = data.positions.reshape(batch, n_electrons, 2)
        return data, means, observables.measure(R, supercell, qs, DENSITY_BINS, r_max, RADIAL_BINS)

    # 4. Burn in, check, measure.
    key = jax.random.PRNGKey(args.seed)
    started = time.time()
    for _ in range(args.burn_in):
        key, subkey = jax.random.split(key)
        data, _, _ = evaluation_step(params, data, subkey)
    print(f"burn-in of {args.burn_in} steps done after {time.time() - started:.0f} s (includes compiling)")

    training_energy = np.loadtxt(run / "train_stats.csv", delimiter=",", skiprows=1, usecols=1)
    summary = dict(run=str(run), checkpoint=final.name, authors_command_line=trained_with["authors_command_line"],
                   laplacian=laplacian or "folx", steps_requested=args.steps, burn_in=args.burn_in,
                   seed=args.seed, batch_size=batch, n_electrons=n_electrons,
                   mcmc_moves_per_step=cfg.mcmc.steps, mcmc_width=float(width),
                   supercell=supercell.tolist(), density_bins=DENSITY_BINS,
                   r_max=float(r_max), radial_bins=RADIAL_BINS,
                   q_vectors=qs.tolist(), q_indices=q_indices.tolist())

    rows, totals = [], {}
    timer = time.time()
    for step in range(args.steps):
        key, subkey = jax.random.split(key)
        data, means, sums = evaluation_step(params, data, subkey)
        rows.append(dict(step=step, **{name: float(value) for name, value in means.items()}))
        add_to(totals, jax.device_get(sums))

        if step + 1 == CHECK_STEPS:
            evaluated = np.mean([row["energy"] for row in rows])
            trained = float(np.mean(training_energy[-CHECK_TRAINING_STEPS:]))
            summary["consistency_check"] = dict(evaluation_first_steps=evaluated,
                                                training_last_steps=trained,
                                                tolerance=CHECK_TOLERANCE)
            print(f"consistency check: evaluation {evaluated:.5f}, end of training {trained:.5f}")
            if abs(evaluated - trained) > CHECK_TOLERANCE:
                save(out, rows, totals, summary)
                sys.exit("consistency check FAILED: the evaluated wave function is not the trained one")

        if (step + 1) % SAVE_EVERY == 0 or step + 1 == args.steps:
            elapsed = time.time() - timer
            summary.update(steps_done=step + 1, n_configurations=(step + 1) * batch,
                           seconds=elapsed, steps_per_second=(step + 1) / elapsed)
            save(out, rows, totals, summary)
            print(f"step {step + 1:6d}  mean energy {np.mean([r['energy'] for r in rows]):.5f}  "
                  f"pmove {rows[-1]['pmove']:.3f}  {(step + 1) / elapsed:.2f} steps/s")


if __name__ == "__main__":
    main()
