"""
Stage 8 -- evaluate a trained network with its weights FROZEN.

Run (one GPU, one evaluation):
    CUDA_VISIBLE_DEVICES=0 python evaluate_weights.py ../../../runs/stage08/training/psiformer_eps5
Options: --steps N     evaluation steps after burn-in (default 10000)
         --burn-in N   steps that only move the walkers, first (default 200)
         --out DIR     default runs/stage08/evaluation/<run name>
         --overwrite   replace an evaluation that is already in DIR

Why a separate evaluation.  The energies in train_stats.csv were measured
while the parameters were still changing, one small update per iteration,
so each number belongs to a slightly different wave function.  Here the
parameters are fixed at the final checkpoint and only the walkers move.
That gives the energy of ONE wave function -- the final one -- as a plain
Monte Carlo average with a clean error bar, and on the same samples the
quantities the paper's Fig. 6 shows (observables.py).  The final wave
function and the average over the end of training need not agree within
the Monte Carlo error bars: they are different wave functions.

What is the authors' and what is ours.  The wave function, the sampler and
the local energy are the authors' own code, built exactly as their train.py
builds them.  The configuration is rebuilt from the run's overrides.json and
must match the config.json saved when it was trained, or the evaluation
stops.  We add only:
  - the move width of the walkers is held FIXED at its final training value
    (training keeps adapting it; with a fixed width the sampler is a
    textbook Metropolis chain whose stationary distribution is |Psi|^2);
  - the energy split into kinetic, moire and Coulomb parts.  The two
    potential terms come from the same functions of pbc/hamiltonian.py
    that the authors' local energy calls; kinetic = total - potentials;
  - the observables of observables.py, measured after every step.
Like the training, everything runs in single precision (float32).

A sanity check before the long run: the mean energy of the first 200
evaluation steps must be finite and within 0.05 (code units, ~0.03 meV per
electron) of the end of training.  A wrong network, wrong weights or a wrong
Hamiltonian would typically miss by far more.  It is a sanity check, not a
proof: different wave functions can have similar energies.  It assumes a
converged run.  While the energy still falls quickly, training reports it
too low -- most likely because the walkers lag behind the changing wave
function and the weights were just fitted to samples like the ones
measured (not tested separately).  A 1600-iteration test run's training
energy was 0.08 below its frozen final weights, and it failed the check.

Safety, as for the training: only one process may evaluate into a folder
(a lock on <out>/.evaluation.lock), and only one job may use a GPU (the lock
runs/stage08/gpu_locks/gpuN.lock, which the training queues of launch_fig5.sh
hold too -- two processes on one GPU crashed the machine on 2026-10-01).  A
refused start exits with code 75.  A finished evaluation is not repeated and
never replaced without --overwrite; "the same evaluation" means the same
checkpoint (by its md5 checksum), steps, burn-in and seed, with all three
output files readable and holding every requested step.  There is no
resuming: an evaluation takes minutes.

Output, in the --out folder (rewritten every 1000 steps):
    energies.csv      one row per step: batch means of E_loc and its parts,
                      the batch variance of E_loc, the acceptance pmove, and
                      -- to see whether the spatial pattern is stationary --
                      S(kappa) and the 9 site occupations of that step
    observables.npz   histograms and sums, added up over all steps
    summary.json      settings, checkpoint, timing, the sanity check
All three record the number of steps they contain; plot_weights.py refuses
files that disagree (as after a crash between two of the writes).
"""
import argparse
import csv
import fcntl
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RUNS = HERE.parents[2] / "runs" / "stage08"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))      # run_moire.py, one folder up

DENSITY_BINS = 90              # 30 bins per moire cell along each supercell vector
RADIAL_BINS = 150
CHECK_STEPS = 200              # evaluation steps used in the sanity check
CHECK_TRAINING_STEPS = 2000    # last training steps it is compared with
CHECK_TOLERANCE = 0.05         # code units of energy (total, all 6 electrons)
SAVE_EVERY = 1000
ALREADY_RUNNING = 75           # exit code: another process evaluates into this folder
KAPPA = 4 * np.pi / 3          # corner of the moire Brillouin zone (a_M = 1)


# ------------------------------------------------- before JAX: folder safety

def hold_folder_lock(folder):
    """Take the exclusive lock on <folder>/.evaluation.lock, or stop with code 75.

    The returned file must stay open while the evaluation runs; the
    operating system releases the lock when the process ends.
    """
    handle = open(folder / ".evaluation.lock", "w")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(f"refusing to start: another process is evaluating into {folder}", file=sys.stderr)
        sys.exit(ALREADY_RUNNING)
    return handle


def existing_evaluation(folder, wanted):
    """None if the folder holds no evaluation, 'finished' if it holds this one, complete, else 'other'.

    `wanted` holds what identifies an evaluation: the checkpoint's md5
    checksum (so a longer training run, or a moved folder, is recognised
    correctly), the steps requested, the burn-in and the seed.  Complete
    means all three files hold the requested number of steps: an overwrite
    stopped between two file writes leaves them disagreeing.
    """
    summary_file = folder / "summary.json"
    if not summary_file.exists():
        return None
    try:                       # a missing, damaged or half-written file counts as unfinished
        summary = json.loads(summary_file.read_text())
        with open(folder / "energies.csv", newline="") as handle:
            csv_steps = sum(1 for _ in csv.DictReader(handle))      # one row per step
        with np.load(folder / "observables.npz") as observables_file:
            npz_steps = int(observables_file["steps_done"])
    except Exception:
        return "other"
    same = all(summary.get(name) == value for name, value in wanted.items())
    complete = csv_steps == npz_steps == summary.get("steps_done") == summary.get("steps_requested")
    return "finished" if same and complete else "other"


def hold_gpu_lock():
    """Take runs/stage08/gpu_locks/gpuN.lock for the one GPU in CUDA_VISIBLE_DEVICES, or stop.

    The training queues of launch_fig5.sh hold the same file with `flock`,
    the same operating-system lock as fcntl.flock here, so an evaluation never
    starts on a GPU that a training run or another evaluation is using.
    """
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not gpu.isdigit():
        sys.exit("make exactly one GPU visible, e.g. CUDA_VISIBLE_DEVICES=0")
    (RUNS / "gpu_locks").mkdir(parents=True, exist_ok=True)
    handle = open(RUNS / "gpu_locks" / f"gpu{gpu}.lock", "w")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(f"refusing to start: GPU {gpu} is in use by another job", file=sys.stderr)
        sys.exit(ALREADY_RUNNING)
    return handle


# ------------------------------------------------- the trained wave function

def apply_overrides(cfg, overrides):
    """Set each 'a.b.c': value from overrides.json on the configuration, as run_moire.py did."""
    for dotted_name, value in overrides.items():
        *path, last = dotted_name.split(".")
        node = cfg
        for key in path:
            node = node[key]
        node[last] = value


def training_configuration(run):
    """The configuration the run was trained with.

    Rebuilt the way run_moire.py built it -- the authors' script, then the
    saved overrides -- and compared with the config.json written at the
    start of training.  If the authors' script or our code has changed
    since, the two differ and we stop rather than evaluate something else.
    """
    from run_moire import authors_configuration
    from periodicwave.utils import custom_logging

    saved = json.loads((run / "overrides.json").read_text())
    cfg = authors_configuration(saved["authors_command_line"])
    apply_overrides(cfg, saved["overrides"])
    with tempfile.TemporaryDirectory() as scratch:
        custom_logging.save_config_dict_as_json(cfg, os.path.join(scratch, "config.json"))
        rebuilt = json.loads(Path(scratch, "config.json").read_text())
    recorded = json.loads((run / "config.json").read_text())
    if rebuilt != recorded:
        differing = sorted(k for k in set(rebuilt) | set(recorded) if rebuilt.get(k) != recorded.get(k))
        sys.exit(f"the rebuilt configuration differs from {run / 'config.json'} in {differing}")
    return cfg, saved


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
    """True if two parameter trees have the same layout and every array the same shape."""
    import jax
    leaves_a, layout_a = jax.tree_util.tree_flatten(a)
    leaves_b, layout_b = jax.tree_util.tree_flatten(b)
    return layout_a == layout_b and all(np.shape(x) == np.shape(y) for x, y in zip(leaves_a, leaves_b))


def file_md5(path):
    with open(path, "rb") as handle:
        return hashlib.md5(handle.read()).hexdigest()


# ------------------------------------------------------------------- output

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
    """Write the three output files.  Each says how many steps it contains."""
    steps_done = len(rows)
    def write_csv(path):
        with open(path, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    def write_npz(path):
        with open(path, "wb") as handle:        # a file handle: np.savez would add ".npz" to a name
            np.savez(handle, steps_done=steps_done, **totals)
    write_atomically(out / "energies.csv", write_csv)
    write_atomically(out / "observables.npz", write_npz)
    write_atomically(out / "summary.json", lambda p: p.write_text(json.dumps(summary, indent=2)))


# ---------------------------------------------------------------------- run

def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("run", type=Path, help="training folder, e.g. runs/stage08/training/psiformer_eps5")
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--burn-in", type=int, default=200)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run = args.run.resolve()
    out = (args.out or RUNS / "evaluation" / run.name).resolve()
    out.mkdir(parents=True, exist_ok=True)

    # 1. Folder and GPU safety, before anything imports JAX or touches a GPU.
    #    The final checkpoint is the last step of the iterations recorded in config.json.
    lock = hold_folder_lock(out)                                   # noqa: F841 -- held until exit
    final_step = json.loads((run / "config.json").read_text())["optim"]["iterations"] - 1
    final = run / f"qmcjax_ckpt_{final_step:06d}.npz"
    if not final.exists():
        sys.exit(f"{final} does not exist: has the run finished?")
    checkpoint_md5 = file_md5(final)
    wanted = dict(checkpoint_md5=checkpoint_md5, steps_requested=args.steps,
                  burn_in=args.burn_in, seed=args.seed)
    existing = existing_evaluation(out, wanted)
    if existing == "finished" and not args.overwrite:
        print(f"{out} already holds this evaluation; use --overwrite to repeat it")
        return
    if existing == "other" and not args.overwrite:
        sys.exit(f"{out} holds a different or unfinished evaluation; use --overwrite to replace it")
    gpu_lock = hold_gpu_lock()                                     # noqa: F841 -- held until exit

    import jax
    import jax.numpy as jnp
    from periodicwave import checkpoint, mcmc, networks
    import observables

    # 2. The configuration, network and weights the run was trained with.
    #    (training_configuration checks the whole configuration against config.json.)
    cfg, trained_with = training_configuration(run)
    network, local_energy, charges = build_like_train_py(cfg)
    if jax.device_count() != 1:
        sys.exit("make exactly one GPU visible (CUDA_VISIBLE_DEVICES=0): "
                 "the checkpoint was saved from one device")
    completed, data, params, _, width = checkpoint.restore(str(final), cfg.batch_size)
    if completed != final_step + 1:
        sys.exit(f"{final.name} holds step {completed - 1}, not {final_step}")
    params = jax.tree_util.tree_map(lambda x: jnp.asarray(x)[0], params)      # drop the device axis
    data = jax.tree_util.tree_map(lambda x: jnp.asarray(x)[0], data)
    width = jnp.asarray(width)[0]
    if not same_structure(params, network.init(jax.random.PRNGKey(0))):
        sys.exit(f"{final.name} does not fit the network built from the configuration")

    # 3. One step: move the walkers (all the burn-in does), then measure.
    batch = cfg.batch_size
    n_electrons = sum(cfg.system.electrons)
    supercell = np.asarray(cfg.system.make_local_energy_kwargs["lattice"])
    g = 4 * np.pi / np.sqrt(3)                                    # |moire reciprocal vector|, a_M = 1
    qs, q_indices = observables.supercell_q_vectors(supercell, q_max=2 * g)
    at_kappa = np.isclose(np.linalg.norm(qs, axis=1), KAPPA, atol=1e-6)
    r_max = 0.5 * np.linalg.norm(supercell[:, 0])                 # g(r) is valid up to half the cell size

    log_abs_psi = lambda p, x, spins, atoms, charges: network.apply(p, x, spins, atoms, charges)[1]  # noqa: E731
    mcmc_step = mcmc.make_mcmc_step(jax.vmap(log_abs_psi, in_axes=(None, 0, 0, 0, 0)), batch,
                                    steps=cfg.mcmc.steps, ndim=cfg.system.ndim, blocks=cfg.mcmc.blocks)
    walker_axes = networks.FermiNetData(positions=0, spins=0, atoms=0, charges=0)
    batch_local_energy = jax.vmap(lambda p, d: local_energy(p, None, d)[0], in_axes=(None, walker_axes))
    batch_potentials = jax.vmap(potential_terms(cfg, charges), in_axes=(walker_axes,))

    @jax.jit
    def move_walkers(params, data, key):
        return mcmc_step(params, data, key, width)[0]

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
        data = move_walkers(params, data, subkey)
    print(f"burn-in of {args.burn_in} steps done after {time.time() - started:.0f} s (includes compiling)")

    training_energy = np.loadtxt(run / "train_stats.csv", delimiter=",", skiprows=1, usecols=1)
    summary = dict(run=str(run), checkpoint=final.name, checkpoint_step=final_step,
                   checkpoint_md5=checkpoint_md5,
                   authors_command_line=trained_with["authors_command_line"],
                   training_overrides=trained_with["overrides"],
                   steps_requested=args.steps, burn_in=args.burn_in,
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
        sums = jax.device_get(sums)
        add_to(totals, sums)
        # Per-step S(kappa) and site occupations: is the spatial pattern stationary?
        s_kappa = float(np.mean(sums["rho_q_squared"][at_kappa]) / (batch * n_electrons))
        occupations = {f"occupation_{k}": float(n) / batch for k, n in enumerate(sums["occupation"])}
        rows.append(dict(step=step, **{name: float(value) for name, value in means.items()},
                         S_kappa=s_kappa, **occupations))

        if step + 1 == CHECK_STEPS:
            evaluated = float(np.mean([row["energy"] for row in rows]))
            trained = float(np.mean(training_energy[-CHECK_TRAINING_STEPS:]))
            summary["sanity_check"] = dict(evaluation_first_steps=evaluated,
                                           training_last_steps=trained, tolerance=CHECK_TOLERANCE)
            print(f"sanity check: evaluation {evaluated:.5f}, end of training {trained:.5f}")
            if not np.isfinite(evaluated) or abs(evaluated - trained) > CHECK_TOLERANCE:
                summary["sanity_check"]["passed"] = False
                save(out, rows, totals, summary)
                sys.exit(f"sanity check FAILED: the first {CHECK_STEPS} steps give {evaluated:.5f}, the "
                         f"end of training {trained:.5f}; not finite, or more than {CHECK_TOLERANCE} apart. "
                         "Either this is not the trained wave function, or the run had not converged "
                         "(while the energy still falls quickly, training reports it too low).")
            summary["sanity_check"]["passed"] = True

        if (step + 1) % SAVE_EVERY == 0 or step + 1 == args.steps:
            elapsed = time.time() - timer
            summary.update(steps_done=step + 1, n_configurations=(step + 1) * batch,
                           seconds=elapsed, steps_per_second=(step + 1) / elapsed)
            save(out, rows, totals, summary)
            print(f"step {step + 1:6d}  mean energy {np.mean([r['energy'] for r in rows]):.5f}  "
                  f"pmove {rows[-1]['pmove']:.3f}  {(step + 1) / elapsed:.2f} steps/s")


if __name__ == "__main__":
    main()
