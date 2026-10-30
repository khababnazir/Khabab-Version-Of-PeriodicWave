"""
Stage 8 -- train one network on the paper's moire system, with OUR code.

Run (one GPU per run), from learn/stage08_own_code:
    CUDA_VISIBLE_DEVICES=0 python train_moire.py --network slaternet --eps 10 \
        --out ../../runs/stage08/training/slaternet_eps10

Options:  --network slaternet | attention     --eps 5 | 10
          --electrons 6 | 18    (default 6: 9 sites, Fig. 5;  18: 27 sites, Table I)
          --settings released | table2    (default released, see below)
          --laplacian hessian | folx      (default hessian;  folx for 18 electrons)
          --determinants det | slogdet    (default det;  slogdet for 18 electrons)
          --learning-rate ETA0    (default: that of --settings)
          --iterations N    (default 200,000 released, 150,000 table2)
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

With --settings released (the default), all settings are those of the
reference run (the authors' released moire config); test_train_moire.py
checks them against its config.json.

18 electrons (added 2026-10-30).  --settings table2 is the paper's Table II
(p. 7): 4096 walkers, 150,000 iterations, learning rate 10 / (1 + t/100000),
and an attention network of 3 layers, 6 heads and 1 perceptron per layer,
built in the released network family that test_networks.py proves equal to
the authors' code.  Table II gives no SlaterNet sizes, so SlaterNet keeps
the released ones, and what Table II does not list keeps its released
value.  At 18 electrons use --laplacian folx --determinants slogdet: Stage
4's full Hessian with Stage 3's det needs too much memory there
(speed_study/speed_study_report.pdf).  The defaults reproduce every earlier
run exactly, and folders made before these options existed still count as
the same experiment (test_train_moire.py checks both).

Output in --out:
    settings.json           what this run is; checked again on every restart
    train_stats.csv         one row per iteration.  Its last two columns come
                            from KFAC: learning_rate, and fisher_step_sq =
                            eta^2 g^T (F + damping)^-1 g, the squared size of
                            the step it proposes.  The step taken has
                            min(fisher_step_sq, norm_constraint): above 1e-3
                            the norm constraint sets the step, not eta.
    checkpoint_NNNNNN.npz   after NNNNNN iterations: parameters, optimiser
                            state, walkers, step width, random key.  Every 30
                            minutes and at the end; never written if any of
                            them is not finite.
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
    # Added for 18 electrons (2026-10-30).  These defaults are what every earlier run used.
    attention_layers: int = 4          # attention network sizes
    attention_heads: int = 4
    attention_perceptrons: int = 2     # perceptrons per layer
    laplacian: str = "hessian"         # kinetic energy: "hessian" (Stage 4) or "folx"
    determinants: str = "det"          # adding the determinants: "det" (Stage 3) or "slogdet"


# The paper's Table II (p. 7).  What it does not list keeps the released value.
TABLE_II = dataclasses.replace(Settings(), batch=4096, learning_rate=10.0, learning_rate_delay=1e5,
                               attention_layers=3, attention_heads=6, attention_perceptrons=1)
SETTINGS = {"released": Settings(), "table2": TABLE_II}
ITERATIONS = {"released": 200000, "table2": 150000}
SITES = {6: 9, 18: 27}                 # electrons -> sites at filling 2/3: Fig. 5 and Table I

# Folders made before 2026-10-30 lack these entries in settings.json; their runs used these values.
BEFORE_18_ELECTRONS = dict(attention_layers=4, attention_heads=4, attention_perceptrons=2,
                           laplacian="hessian", determinants="det")


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


def make_problem(network, eps, electrons=6, settings=Settings()):
    """The wave function and local energy: 6 electrons on 9 sites, or 18 on 27."""
    import slaternet
    import psiformer
    from local_energy import local_energy, kinetic_energy_folx, moire_potential
    from ewald import make_ewald_setup, coulomb_energy

    lattice = slaternet.supercell_vectors(SITES[electrons])
    G = slaternet.reciprocal_vectors(lattice)
    if network == "slaternet":
        arch = slaternet.ReleasedArchitecture(n_electrons=electrons, determinants=settings.determinants)
        init, log_psi_released = slaternet.init_released, slaternet.log_psi_released
    elif network == "attention":
        arch = psiformer.ReleasedAttentionArchitecture(
            n_electrons=electrons, n_layers=settings.attention_layers, n_heads=settings.attention_heads,
            perceptrons_per_layer=settings.attention_perceptrons, determinants=settings.determinants)
        init, log_psi_released = psiformer.init_released, psiformer.log_psi_released
    else:
        raise ValueError(f"unknown network {network!r}")
    if settings.laplacian not in ("hessian", "folx"):
        raise ValueError(f"laplacian must be 'hessian' or 'folx', not {settings.laplacian!r}")

    V, U, phi = moire_constants(eps)
    ewald = make_ewald_setup(lattice)

    def log_psi(params, R):
        return log_psi_released(params, R, G, arch)

    def e_loc(params, R):
        psi = lambda X: log_psi(params, X)                 # noqa: E731
        if settings.laplacian == "folx":                   # the same formula, computed faster (Stage 4)
            kinetic_and_moire = kinetic_energy_folx(psi, R) + moire_potential(R, V, phi)
        else:
            kinetic_and_moire = local_energy(psi, R, V, phi)
        return kinetic_and_moire + coulomb_energy(ewald, R, strength=U)

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

def all_finite(tree):
    """True if no number in tree is NaN or inf (integer arrays always pass)."""
    import jax
    arrays = [np.asarray(leaf) for leaf in jax.tree.leaves(tree)]
    return all(np.isfinite(a).all() for a in arrays if np.issubdtype(a.dtype, np.inexact))


def save_checkpoint(folder, done, params, opt_state, walkers, width, key):
    """checkpoint_<done>.npz, written via a temporary file so it is never half written.

    Stops the run instead of saving if the parameters, the optimiser state or
    the walkers hold a NaN or inf: the newest checkpoint must always be one
    a run can continue from.
    """
    import jax
    for name, tree in (("parameters", params), ("optimiser state", opt_state), ("walkers", walkers)):
        if not all_finite(tree):
            sys.exit(f"after {done} iterations the {name} are not finite; stopping without "
                     "saving (the last checkpoint is unchanged)")
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


def restore_tree(saved, like, prefix, path):
    """The arrays prefix_0, prefix_1, ... of an open checkpoint, in the structure of `like`."""
    import jax
    import jax.numpy as jnp
    leaves, structure = jax.tree.flatten(like)
    names = [f"{prefix}_{i}" for i in range(len(leaves))]
    if (any(name not in saved.files for name in names) or f"{prefix}_{len(leaves)}" in saved.files
            or any(saved[name].shape != np.shape(leaf) for name, leaf in zip(names, leaves))):
        raise ValueError(f"{path.name} does not fit this network or optimiser")
    return jax.tree.unflatten(structure, [jnp.asarray(saved[name], dtype=jnp.asarray(leaf).dtype)
                                          for name, leaf in zip(names, leaves)])


def load_checkpoint(path, params_like, opt_state_like):
    """The arrays of a checkpoint, put back into the structure of params_like / opt_state_like."""
    import jax.numpy as jnp
    with np.load(path) as saved:
        return (int(saved["done"]), restore_tree(saved, params_like, "params", path),
                restore_tree(saved, opt_state_like, "opt_state", path),
                jnp.asarray(saved["walkers"]), float(saved["width"]), jnp.asarray(saved["key"]))


def load_parameters(path, params_like):
    """What an evaluation needs from a checkpoint: iterations done, parameters, walkers, step width."""
    import jax.numpy as jnp
    with np.load(path) as saved:
        return (int(saved["done"]), restore_tree(saved, params_like, "params", path),
                jnp.asarray(saved["walkers"]), float(saved["width"]))


def checkpoints_newest_first(folder):
    return sorted(folder.glob("checkpoint_*.npz"), reverse=True)


def is_finished(folder, iterations):
    """True if the final checkpoint exists, reads completely and says it holds
    `iterations`.  Its file name alone is not enough: a damaged final
    checkpoint must not make a run count as finished."""
    try:
        with np.load(folder / f"checkpoint_{iterations:06d}.npz") as saved:
            for name in saved.files:
                _ = saved[name]                       # reading every array checks the whole file
            return int(saved["done"]) == iterations
    except Exception:                                  # missing, damaged or incomplete
        return False


# ------------------------------------------------------------- training

STATS_COLUMNS = ["step", "energy", "variance", "energy_imag", "pmove", "step_width", "seconds",
                 "learning_rate", "fisher_step_sq"]


def train(folder, network, eps, gradient, seed, iterations, settings=Settings(), log=print, electrons=6):
    """Train, or continue training, into folder.  Returns the parameters."""
    import jax
    import jax.numpy as jnp
    from metropolis import adapt_step_width

    problem = make_problem(network, eps, electrons, settings)
    move, sample_and_measure = make_sampler(problem, settings, gradient)
    optimizer = make_optimizer(make_loss(problem, gradient), settings)

    # A fresh start.  Random numbers: one key each for parameters, walkers, everything after.
    key = jax.random.PRNGKey(seed)
    key, key_params, key_walkers = jax.random.split(key, 3)
    params = problem.init(key_params, problem.arch)
    walkers = settings.init_width * jax.random.normal(key_walkers, (settings.batch, electrons, 2))
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
    capped = window = 0        # iterations, since the last log line, in which the norm constraint set the step
    with open(stats_file, "a", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        for step in range(done, iterations):
            key, key_move, key_kfac = jax.random.split(key, 3)
            walkers, e_for_gradient, stats = sample_and_measure(params, walkers, key_move, width)
            params, opt_state, kfac = optimizer.step(
                params, opt_state, key_kfac, batch=(walkers, e_for_gradient),
                momentum=jnp.asarray(0.0), damping=jnp.asarray(settings.damping))

            stats = {name: float(value) for name, value in stats.items()}
            if not np.isfinite(stats["energy"]):
                sys.exit(f"step {step}: the energy is not finite; stopping without saving "
                         "(the last checkpoint is unchanged)")
            learning_rate, fisher_step_sq = float(kfac["learning_rate"]), float(kfac["scaled_grad_norm_sq"])
            capped, window = capped + (fisher_step_sq > settings.norm_constraint), window + 1
            writer.writerow([step, stats["energy"], stats["variance"], stats["energy_imag"],
                             stats["pmove"], width, round(time.time() - started, 3),
                             learning_rate, fisher_step_sq])
            if step % settings.adapt_every == 0:              # after the step, as the authors do
                width = adapt_step_width(width, stats["pmove"])

            if (step + 1) % 1000 == 0:
                rate = (step + 1 - done) / (time.time() - started)
                log(f"step {step + 1:7d}  energy {stats['energy']:+.5f}  pmove {stats['pmove']:.2f}  "
                    f"width {width:.3f}  lr {learning_rate:.3g}  step capped {100 * capped / window:.0f}%  "
                    f"{rate:.2f} it/s")
                capped = window = 0
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
    # Folders made before 2026-10-30 lack the 18-electron entries: 6 electrons and BEFORE_18_ELECTRONS.
    saved.setdefault("electrons", 6)
    saved["settings"] = {**BEFORE_18_ELECTRONS, **saved["settings"]}
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
    parser.add_argument("--electrons", type=int, choices=sorted(SITES), default=6)
    parser.add_argument("--settings", choices=sorted(SETTINGS), default="released")
    parser.add_argument("--laplacian", choices=["hessian", "folx"], default="hessian")
    parser.add_argument("--determinants", choices=["det", "slogdet"], default="det")
    parser.add_argument("--learning-rate", type=float, default=None, help="default: that of --settings")
    parser.add_argument("--iterations", type=int, default=None,
                        help="default 200000 (released), 150000 (table2)")
    parser.add_argument("--gradient", choices=["real", "full"], default="real")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    settings = dataclasses.replace(SETTINGS[args.settings], laplacian=args.laplacian,
                                   determinants=args.determinants)
    if args.learning_rate is not None:
        settings = dataclasses.replace(settings, learning_rate=args.learning_rate)
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
    # If CUDA cannot start, JAX quietly falls back to the CPU and the run would
    # crawl on unnoticed (it happened on 2026-10-29, after GPU 1 failed).
    if jax.default_backend() != "gpu":
        sys.exit(f"JAX found no working GPU (backend: {jax.default_backend()}); not starting")
    wanted = dict(network=args.network, eps=args.eps, electrons=args.electrons, gradient=args.gradient,
                  seed=args.seed,
                  iterations=ITERATIONS[args.settings] if args.iterations is None else args.iterations,
                  settings=dataclasses.asdict(settings))
    iterations = same_experiment_or_stop(folder, wanted, args.iterations)
    if is_finished(folder, iterations):
        print(f"{folder} has already finished: it holds checkpoint_{iterations:06d}.npz")
        return
    import folx
    with open(folder / "starts.log", "a") as handle:
        handle.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  GPU {gpu}  target {iterations}  "
                     f"jax {jax.__version__}  kfac_jax {getattr(kfac_jax, '__version__', '?')}  "
                     f"folx {getattr(folx, '__version__', '?')}\n")

    # Matrix products in full single precision.  On these GPUs JAX would
    # otherwise use TF32, which keeps only about 3 digits.  The authors' config
    # sets the same.
    jax.config.update("jax_default_matmul_precision", "float32")
    train(folder, args.network, args.eps, args.gradient, args.seed, iterations, settings,
          electrons=args.electrons)
    print("finished")


if __name__ == "__main__":
    main()
