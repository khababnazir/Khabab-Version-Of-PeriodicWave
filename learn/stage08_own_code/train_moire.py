"""
Stage 8 -- train one network on the paper's moire system, with OUR code.

Run (one GPU per run), from learn/stage08_own_code:
    CUDA_VISIBLE_DEVICES=0 python train_moire.py --network slaternet --eps 10 \
        --out ../../runs/stage08/training/slaternet_eps10

Options:  --network slaternet | attention     --eps 5 | 10
          --iterations N    (default 200,000, as the reference run)
          --gradient real | full    (default real, see below)
          --seed S          (default 0)

Which code does what.  Everything is ours except the KFAC library:
    wave function   Stage 3 / Stage 5, their "released settings" sections
    sampling        Stage 4: metropolis_step, adapt_step_width
    local energy    Stage 4: kinetic + moire;  Stage 7: Coulomb (Ewald)
    gradient        Stage 6: clip_local_energy, and the formula of energy_gradient
    optimiser       kfac-jax, with the authors' settings and layer patterns

One iteration, in the order of the authors' train.py:
    1. move every walker by 30 Metropolis steps        walkers ~ |Psi|^2
    2. local energy E_loc at every walker              energy = mean of Re E_loc
    3. clip E_loc for the gradient (Stage 6)
    4. one KFAC step with the gradient 2 Re < (E - <E>)* O >  (E clipped)
    5. every 100 iterations, adapt the Metropolis step width

The gradient.  --gradient real keeps only Re E_loc in the gradient, as the
authors' code does (Stage 6 found that their folx path drops Im E_loc).  The
reference run was trained that way, so it is the default.  --gradient full
keeps Im E_loc (Stage 6's exact formula).  The reported energy is the mean of
Re E_loc either way.

All settings are those of the reference run (the authors' released moire
config); test_train_moire.py checks them against its config.json.

Output in --out:
    settings.json           what this run is; checked again on every restart
    train_stats.csv         one row per iteration
    checkpoint_NNNNNN.npz   after NNNNNN iterations: parameters, optimiser
                            state, walkers, step width, random key.  Every 30
                            minutes and at the end.
    starts.log              one line per start

Restarting: run the same command again.  It continues from the newest
checkpoint, exactly as if it had not stopped.  Safety, checked before JAX
touches a GPU: one process per folder, one job per GPU (runs/gpu_locks,
shared with every other stage), a restart must be the same experiment, and a
finished run exits at once.  A refused start exits with code 75.
"""
import argparse
import csv
import dataclasses
import fcntl
import json
import os
import sys
import time
from pathlib import Path
from typing import NamedTuple

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
GPU_LOCKS = REPO / "runs" / "gpu_locks"
for stage in ("stage03_slater", "stage04_mcmc", "stage05_attention",
              "stage06_natural_gradient", "stage07_ewald"):
    sys.path.insert(0, str(REPO / "learn" / stage))

ALREADY_RUNNING = 75           # exit code: another process holds this folder or GPU
SAVE_EVERY_SECONDS = 30 * 60   # as the authors' save_frequency


@dataclasses.dataclass(frozen=True)
class Settings:
    """The reference run's settings (authors' released moire config)."""
    batch: int = 1024                  # walkers
    mcmc_steps: int = 30               # Metropolis steps per iteration
    burn_in: int = 300                 # iterations of mcmc_steps moves before training
    init_width: float = 1.0            # initial positions: normal * 1.0 around the origin
    move_width: float = 0.2            # initial Metropolis step width
    adapt_every: int = 100             # adapt the step width every 100 iterations
    learning_rate: float = 0.1         # 0.1 / (1 + t / 20000)
    learning_rate_delay: float = 20000.0
    learning_rate_decay: float = 1.0
    damping: float = 1e-3              # KFAC
    norm_constraint: float = 1e-3      # KFAC: limit on the size of a step
    curvature_ema: float = 0.95        # KFAC: running average of the curvature
    min_damping: float = 1e-6          # KFAC
    clip_scale: float = 5.0            # gradient-only clipping of E_loc (Stage 6)


# ------------------------------------------------------------- the physics

class Problem(NamedTuple):
    arch: object          # network sizes
    init: object          # init(key, arch) -> parameters
    log_psi: object       # log_psi(params, R) -> LogPsi, R of shape (N, 2)
    local_energy: object  # local_energy(params, R) -> complex E_loc


def moire_constants(eps):
    """V (15 meV), U (Coulomb strength at this eps) and phi, in code units (Stages 4 and 7)."""
    from local_energy import moire_units
    from ewald import moire_interaction_strength
    _, V = moire_units()
    return V, moire_interaction_strength(dielectric_constant=eps), np.pi / 4


def make_problem(network, eps):
    """The wave function and local energy: 6 electrons on 9 sites."""
    import slaternet
    import psiformer
    from local_energy import local_energy
    from ewald import make_ewald_setup, coulomb_energy

    lattice = slaternet.supercell_vectors(9)
    G = slaternet.reciprocal_vectors(lattice)
    if network == "slaternet":
        arch = slaternet.ReleasedArchitecture(n_electrons=6)
        init, log_psi_released = slaternet.init_released, slaternet.log_psi_released
    elif network == "attention":
        arch = psiformer.ReleasedAttentionArchitecture(n_electrons=6)
        init, log_psi_released = psiformer.init_released, psiformer.log_psi_released
    else:
        raise ValueError(f"unknown network {network!r}")

    V, U, phi = moire_constants(eps)
    ewald = make_ewald_setup(lattice)

    def log_psi(params, R):
        return log_psi_released(params, R, G, arch)

    def e_loc(params, R):
        psi = lambda X: log_psi(params, X)                 # noqa: E731
        return local_energy(psi, R, V, phi) + coulomb_energy(ewald, R, strength=U)

    return Problem(arch, init, log_psi, e_loc)


# ------------------------------------------------------------- one iteration

def make_sampler(problem, settings, gradient):
    """Steps 1-3 of an iteration, compiled.  Returns two functions:
        move(params, walkers, key, width) -> walkers, acceptance rate   (step 1 only: burn-in)
        sample_and_measure(...)           -> walkers, E for the gradient, statistics
    """
    import jax
    import jax.numpy as jnp
    from metropolis import metropolis_step
    from gradient import clip_local_energy

    def log_abs_batch(params, walkers):
        return jax.vmap(lambda R: problem.log_psi(params, R).log_abs)(walkers)

    @jax.jit
    def move(params, walkers, key, width):
        """mcmc_steps Metropolis steps, all walkers at once (Stage 4)."""
        def one_step(i, state):
            walkers, log_prob, key, accepted = state
            key, step_key = jax.random.split(key)
            walkers, log_prob, rate = metropolis_step(
                lambda W: log_abs_batch(params, W), walkers, log_prob, step_key, width)
            return walkers, log_prob, key, accepted + rate
        start = (walkers, 2.0 * log_abs_batch(params, walkers), key, jnp.zeros(()))
        walkers, _, _, accepted = jax.lax.fori_loop(0, settings.mcmc_steps, one_step, start)
        return walkers, accepted / settings.mcmc_steps

    @jax.jit
    def sample_and_measure(params, walkers, key, width):
        # 1. move the walkers.
        walkers, pmove = move(params, walkers, key, width)

        # 2. local energies.
        e_loc = jax.vmap(lambda R: problem.local_energy(params, R))(walkers)

        # 3. what the gradient sees.
        if gradient == "real":
            e_for_gradient = jnp.real(clip_local_energy(jnp.real(e_loc), settings.clip_scale))
        else:
            e_for_gradient = clip_local_energy(e_loc, settings.clip_scale)

        stats = dict(energy=jnp.mean(jnp.real(e_loc)), variance=jnp.var(jnp.real(e_loc)),
                     energy_imag=jnp.mean(jnp.imag(e_loc)), pmove=pmove)
        return walkers, e_for_gradient, stats

    return move, sample_and_measure


def make_loss(problem, gradient):
    """What KFAC differentiates.  batch = (walkers, clipped E_loc).

    The clipped local energies arrive as data, so jax.grad differentiates
    only log|Psi| and the phase.  With d = E - <E> (clipped energies),

        d loss / d theta = 2 < Re d  d log|Psi|/d theta  +  Im d  d phase/d theta >
                         = 2 Re < d* O >,      O = d log Psi / d theta,

    which is Stage 6's energy_gradient (test_train_moire.py checks it).  The
    loss VALUE is not the energy; with our settings KFAC uses only the
    gradient.  With --gradient real, d is real and the phase term is left out.

    register_normal_predictive_distribution tells KFAC that its curvature
    should be the Fisher matrix of |Psi|^2, as in the authors' loss.py.
    """
    import jax
    import jax.numpy as jnp
    import kfac_jax

    def loss(params, key, batch):
        walkers, e_clipped = batch
        out = jax.vmap(lambda R: problem.log_psi(params, R))(walkers)
        kfac_jax.register_normal_predictive_distribution(out.log_abs[:, None])
        d = e_clipped - jnp.mean(e_clipped)
        value = 2.0 * jnp.mean(jnp.real(d) * out.log_abs)
        if gradient == "full":
            value = value + 2.0 * jnp.mean(jnp.imag(d) * out.phase)
        return value, {}

    return loss


def make_optimizer(loss, settings):
    """kfac-jax with the authors' settings, as their train.py builds it."""
    import jax
    import kfac_jax
    from periodicwave import curvature_tags_and_blocks   # the authors' KFAC layer patterns

    def learning_rate(t):
        return settings.learning_rate * (1.0 / (1.0 + t / settings.learning_rate_delay)) ** settings.learning_rate_decay

    return kfac_jax.Optimizer(
        value_and_grad_func=jax.value_and_grad(loss, has_aux=True),
        l2_reg=0.0, norm_constraint=settings.norm_constraint,
        value_func_has_aux=True, value_func_has_rng=True,
        learning_rate_schedule=learning_rate, curvature_ema=settings.curvature_ema,
        inverse_update_period=1, min_damping=settings.min_damping, num_burnin_steps=0,
        register_only_generic=False, estimation_mode="fisher_exact", multi_device=False,
        auto_register_kwargs=dict(graph_patterns=curvature_tags_and_blocks.GRAPH_PATTERNS))


# ------------------------------------------------------------- checkpoints

def save_checkpoint(folder, done, params, opt_state, walkers, width, key):
    """checkpoint_<done>.npz, written via a temporary file so it is never half written."""
    import jax
    arrays = dict(done=done, walkers=np.asarray(walkers), width=width, key=np.asarray(key))
    for i, leaf in enumerate(jax.tree.leaves(params)):
        arrays[f"params_{i}"] = np.asarray(leaf)
    for i, leaf in enumerate(jax.tree.leaves(opt_state)):
        arrays[f"opt_state_{i}"] = np.asarray(leaf)
    path = folder / f"checkpoint_{done:06d}.npz"
    temporary = folder / f"checkpoint_{done:06d}.tmp"
    with open(temporary, "wb") as handle:          # a handle: np.savez would add ".npz" to a name
        np.savez(handle, **arrays)
    os.replace(temporary, path)


def load_checkpoint(path, params_like, opt_state_like):
    """The arrays of a checkpoint, put back into the structure of params_like / opt_state_like."""
    import jax
    import jax.numpy as jnp
    with np.load(path) as saved:
        def restore(like, prefix):
            leaves, structure = jax.tree.flatten(like)
            stored = [saved[f"{prefix}_{i}"] for i in range(len(leaves))]
            if f"{prefix}_{len(leaves)}" in saved.files or any(a.shape != np.shape(b) for a, b in zip(stored, leaves)):
                raise ValueError(f"{path.name} does not fit this network or optimiser")
            return jax.tree.unflatten(structure, [jnp.asarray(a, dtype=jnp.asarray(b).dtype)
                                                  for a, b in zip(stored, leaves)])
        return (int(saved["done"]), restore(params_like, "params"), restore(opt_state_like, "opt_state"),
                jnp.asarray(saved["walkers"]), float(saved["width"]), jnp.asarray(saved["key"]))


def checkpoints_newest_first(folder):
    return sorted(folder.glob("checkpoint_*.npz"), reverse=True)


# ------------------------------------------------------------- training

STATS_COLUMNS = ["step", "energy", "variance", "energy_imag", "pmove", "step_width", "seconds"]


def train(folder, network, eps, gradient, seed, iterations, settings=Settings(), log=print):
    """Train, or continue training, into folder.  Returns the parameters."""
    import jax
    import jax.numpy as jnp
    from metropolis import adapt_step_width

    problem = make_problem(network, eps)
    move, sample_and_measure = make_sampler(problem, settings, gradient)
    optimizer = make_optimizer(make_loss(problem, gradient), settings)

    # A fresh start.  Random numbers: one key each for parameters, walkers, everything after.
    key = jax.random.PRNGKey(seed)
    key, key_params, key_walkers = jax.random.split(key, 3)
    params = problem.init(key_params, problem.arch)
    walkers = settings.init_width * jax.random.normal(key_walkers, (settings.batch, 6, 2))
    width = settings.move_width
    done = 0
    no_energies = jnp.zeros(settings.batch) + (0j if gradient == "full" else 0.0)
    opt_state = optimizer.init(params, key, (walkers, no_energies))

    # ... or continue from the newest readable checkpoint.
    for path in checkpoints_newest_first(folder):
        try:
            done, params, opt_state, walkers, width, key = load_checkpoint(path, params, opt_state)
            log(f"continuing from {path.name}")
            break
        except Exception as error:                  # damaged or incomplete: try the one before
            log(f"cannot read {path.name} ({error}); trying the previous checkpoint")

    stats_file = folder / "train_stats.csv"
    if done == 0:
        log(f"burn-in: {settings.burn_in} x {settings.mcmc_steps} Metropolis steps")
        for _ in range(settings.burn_in):
            key, key_move = jax.random.split(key)
            walkers, _ = move(params, walkers, key_move, width)
        with open(stats_file, "w", newline="") as handle:
            csv.writer(handle, lineterminator="\n").writerow(STATS_COLUMNS)
    else:
        # Rows written after the checkpoint belong to iterations that are now repeated.
        with open(stats_file, newline="") as handle:
            rows = [row for row in csv.DictReader(handle) if int(row["step"]) < done]
        with open(stats_file, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=STATS_COLUMNS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)

    started = time.time()
    last_save = time.time()
    with open(stats_file, "a", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        for step in range(done, iterations):
            key, key_move, key_kfac = jax.random.split(key, 3)
            walkers, e_for_gradient, stats = sample_and_measure(params, walkers, key_move, width)
            params, opt_state, _ = optimizer.step(
                params, opt_state, key_kfac, batch=(walkers, e_for_gradient),
                momentum=jnp.asarray(0.0), damping=jnp.asarray(settings.damping))

            stats = {name: float(value) for name, value in stats.items()}
            if not np.isfinite(stats["energy"]):
                sys.exit(f"step {step}: the energy is not finite; stopping without saving "
                         "(the last checkpoint is unchanged)")
            writer.writerow([step, stats["energy"], stats["variance"], stats["energy_imag"],
                             stats["pmove"], width, round(time.time() - started, 3)])
            if step % settings.adapt_every == 0:              # after the step, as the authors do
                width = adapt_step_width(width, stats["pmove"])

            if (step + 1) % 1000 == 0:
                rate = (step + 1 - done) / (time.time() - started)
                log(f"step {step + 1:7d}  energy {stats['energy']:+.5f}  pmove {stats['pmove']:.2f}  "
                    f"width {width:.3f}  {rate:.2f} it/s")
            if time.time() - last_save > SAVE_EVERY_SECONDS or step + 1 == iterations:
                handle.flush()
                save_checkpoint(folder, step + 1, params, opt_state, walkers, width, key)
                last_save = time.time()
    return params


# ------------------------------------------------------------- safety and start

def hold_lock(path, what):
    """An exclusive lock, held until the process ends, or stop with code 75."""
    handle = open(path, "w")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(f"refusing to start: {what}", file=sys.stderr)
        sys.exit(ALREADY_RUNNING)
    return handle


def same_experiment_or_stop(folder, wanted, iterations_given):
    """settings.json says what a folder holds.  Returns the iteration target."""
    path = folder / "settings.json"
    if not path.exists():
        path.write_text(json.dumps(wanted, indent=2))
        return wanted["iterations"]
    saved = json.loads(path.read_text())
    differing = [name for name in wanted if name != "iterations" and saved.get(name) != wanted[name]]
    if differing:
        sys.exit(f"{folder} holds a different experiment (differs in {differing}); use another --out")
    if iterations_given is None:
        return saved["iterations"]
    saved["iterations"] = iterations_given
    path.write_text(json.dumps(saved, indent=2))
    return iterations_given


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--network", choices=["slaternet", "attention"], required=True)
    parser.add_argument("--eps", type=float, choices=[5.0, 10.0], required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=None, help="default 200000")
    parser.add_argument("--gradient", choices=["real", "full"], default="real")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    folder = args.out.resolve()
    folder.mkdir(parents=True, exist_ok=True)

    # Safety first, before JAX is imported.
    folder_lock = hold_lock(folder / ".training.lock", f"another process trains into {folder}")  # noqa: F841
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not gpu.isdigit():
        sys.exit("make exactly one GPU visible, e.g. CUDA_VISIBLE_DEVICES=0")
    GPU_LOCKS.mkdir(parents=True, exist_ok=True)
    gpu_lock = hold_lock(GPU_LOCKS / f"gpu{gpu}.lock", f"GPU {gpu} is in use by another job")  # noqa: F841

    import jax
    import kfac_jax
    wanted = dict(network=args.network, eps=args.eps, gradient=args.gradient, seed=args.seed,
                  iterations=200000 if args.iterations is None else args.iterations,
                  settings=dataclasses.asdict(Settings()))
    iterations = same_experiment_or_stop(folder, wanted, args.iterations)
    newest = checkpoints_newest_first(folder)
    if newest and newest[0].name == f"checkpoint_{iterations:06d}.npz":
        print(f"{folder} has already finished: it holds {newest[0].name}")
        return
    with open(folder / "starts.log", "a") as handle:
        handle.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  GPU {gpu}  target {iterations}  "
                     f"jax {jax.__version__}  kfac_jax {getattr(kfac_jax, '__version__', '?')}\n")

    # Matrix products in full single precision.  On these GPUs JAX would
    # otherwise use TF32, which keeps only about 3 digits.  The authors' config
    # sets the same.
    jax.config.update("jax_default_matmul_precision", "float32")
    train(folder, args.network, args.eps, args.gradient, args.seed, iterations)
    print("finished")


if __name__ == "__main__":
    main()
