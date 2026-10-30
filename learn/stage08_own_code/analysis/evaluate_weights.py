"""
Stage 8 (our code) -- evaluate a trained network with its weights FROZEN.

Run (one GPU, one evaluation), from learn/stage08_own_code/analysis:
    CUDA_VISIBLE_DEVICES=0 ../../../.venv/bin/python evaluate_weights.py ../../../runs/stage08/training/attention_eps5
Options: --steps N     evaluation steps after burn-in (default 10000)
         --burn-in N   steps that only move the walkers, first (default 200)
         --seed S      (default 0)
         --out DIR     default runs/stage08/evaluation/<run name>
         --overwrite   replace an evaluation that is already in DIR

The same evaluation as learn/archive/stage08_authors_code/analysis/
evaluate_weights.py, with OUR code where that one uses the authors':
    wave function, local energy    train_moire.make_problem   (Stages 3, 4, 5, 7)
    Metropolis moves               train_moire.make_sampler   (Stage 4), 30 per step
    weights, walkers, step width   the final checkpoint, train_moire.load_parameters
    observables                    observables.py, a copy of the archived one
Settings, output files and sanity check are the same, so the two evaluations
-- and the figures made from them -- can be compared number by number.

Why a separate evaluation.  The energies in train_stats.csv were measured
while the parameters were still changing, one small update per iteration,
so each number belongs to a slightly different wave function.  Here the
parameters are fixed at the final checkpoint and only the walkers move.
That gives the energy of ONE wave function -- the final one -- as a plain
Monte Carlo average with a clean error bar, and on the same samples the
quantities the paper's Fig. 6 shows (observables.py).  The final wave
function and the average over the end of training need not agree within
the Monte Carlo error bars: they are different wave functions.

What is added to the training's code:
  - the step width of the walkers is held FIXED at its final training value
    (training keeps adapting it; with a fixed width the sampler is a
    textbook Metropolis chain whose stationary distribution is |Psi|^2);
  - the energy split into kinetic, moire and Coulomb.  The two potentials
    come from the same functions and constants that the local energy uses
    (potential_parts); kinetic = total - potentials;
  - the observables of observables.py, measured after every step.
Like the training, everything runs in single precision (float32), matrix
products included.

A sanity check before the long run: the mean energy of the first 200
evaluation steps must be finite and within 0.05 (code units, ~0.03 meV per
electron) of the mean of the last 2000 training iterations.  A wrong network,
wrong weights or a wrong Hamiltonian would typically miss by far more.  It is
a sanity check, not a proof: different wave functions can have similar
energies.

Safety, as for the training: only one process may evaluate into a folder
(a lock on <out>/.evaluation.lock), only one job may use a GPU (the lock
runs/gpu_locks/gpuN.lock, shared with the training), and the evaluation
refuses to start without a working GPU.  A refused start exits with code 75.
A finished evaluation is not repeated and never replaced without
--overwrite; "the same evaluation" means the same checkpoint (by its md5
checksum), steps, burn-in and seed, with all three output files readable
and holding every requested step.  There is no resuming: an evaluation
takes minutes.

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
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RUNS = HERE.parents[2] / "runs" / "stage08"
sys.path.insert(0, str(HERE))             # observables.py
sys.path.insert(0, str(HERE.parent))      # train_moire.py, one folder up
import train_moire                        # noqa: E402 -- imports JAX only inside its functions

DENSITY_BINS = 90              # 30 bins per moire cell along each supercell vector
RADIAL_BINS = 150
CHECK_STEPS = 200              # evaluation steps used in the sanity check
CHECK_TRAINING_STEPS = 2000    # last training steps it is compared with
CHECK_TOLERANCE = 0.05         # code units of energy (total, all 6 electrons)
SAVE_EVERY = 1000
KAPPA = 4 * np.pi / 3          # corner of the moire Brillouin zone (a_M = 1)
G = 4 * np.pi / np.sqrt(3)     # |moire reciprocal vector|


# ------------------------------------------------- the trained wave function

def final_checkpoint(run):
    """The run's settings.json, and the path of its final checkpoint."""
    settings = json.loads((run / "settings.json").read_text())
    return settings, run / f"checkpoint_{settings['iterations']:06d}.npz"


def potential_parts(eps):
    """parts(R) -> (moire, Coulomb) energy of one configuration R of shape (N, 2).

    The same functions and constants as the local energy of
    train_moire.make_problem, so local energy - moire - Coulomb is exactly
    the kinetic energy (test_evaluate_weights.py checks it).
    """
    import slaternet
    from local_energy import moire_potential
    from ewald import make_ewald_setup, coulomb_energy

    V, U, phi = train_moire.moire_constants(eps)
    ewald = make_ewald_setup(slaternet.supercell_vectors(9))

    def parts(R):
        return moire_potential(R, V, phi), coulomb_energy(ewald, R, strength=U)
    return parts


def file_md5(path):
    with open(path, "rb") as handle:
        return hashlib.md5(handle.read()).hexdigest()


# ------------------------------------------------------------------- output

def existing_evaluation(folder, wanted):
    """None if the folder holds no evaluation, 'finished' if it holds this one, complete, else 'other'.

    `wanted` holds what identifies an evaluation: the checkpoint's md5
    checksum, the steps requested, the burn-in and the seed.  Complete means
    all three files hold the requested number of steps: an overwrite stopped
    between two file writes leaves them disagreeing.
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
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)

    def write_npz(path):
        with open(path, "wb") as handle:        # a file handle: np.savez would add ".npz" to a name
            np.savez(handle, steps_done=steps_done, **totals)
    write_atomically(out / "energies.csv", write_csv)
    write_atomically(out / "observables.npz", write_npz)
    write_atomically(out / "summary.json", lambda p: p.write_text(json.dumps(summary, indent=2)))


# ---------------------------------------------------------------------- run

def evaluate(run, out, steps, burn_in, seed, check_steps=CHECK_STEPS, log=print):
    """Evaluate the final weights of the training folder `run` into `out`.  Returns the summary."""
    import jax
    import jax.numpy as jnp
    import observables
    import slaternet

    # 1. The network, local energy and Metropolis moves of the training, and its final weights.
    settings, final = final_checkpoint(run)
    if settings.get("electrons", 6) != 6:
        sys.exit(f"{run.name}: this evaluation handles the 9-site runs (6 electrons) only, so far")
    problem = train_moire.make_problem(settings["network"], settings["eps"])
    move, _ = train_moire.make_sampler(problem, train_moire.Settings(), settings["gradient"])
    params_like = problem.init(jax.random.PRNGKey(0), problem.arch)
    done, params, walkers, width = train_moire.load_parameters(final, params_like)
    if done != settings["iterations"]:
        sys.exit(f"{final.name} holds {done} iterations, not {settings['iterations']}")
    batch, n_electrons = walkers.shape[0], walkers.shape[1]

    # 2. What is measured.
    supercell = np.asarray(slaternet.supercell_vectors(9))       # columns L1, L2, as observables.py wants
    qs, q_indices = observables.supercell_q_vectors(supercell, q_max=2 * G)
    at_kappa = np.isclose(np.linalg.norm(qs, axis=1), KAPPA, atol=1e-6)
    r_max = 0.5 * np.linalg.norm(supercell[:, 0])                 # g(r) is valid up to half the cell size
    parts = potential_parts(settings["eps"])

    @jax.jit
    def evaluation_step(params, walkers, key):
        walkers, pmove = move(params, walkers, key, width)        # width: fixed, the final training value
        energy = jnp.real(jax.vmap(lambda R: problem.local_energy(params, R))(walkers))
        moire, coulomb = jax.vmap(parts)(walkers)
        means = dict(energy=energy.mean(), variance=energy.var(),
                     kinetic=(energy - moire - coulomb).mean(),
                     moire=moire.mean(), coulomb=coulomb.mean(), pmove=pmove)
        return walkers, means, observables.measure(walkers, supercell, qs, DENSITY_BINS, r_max, RADIAL_BINS)

    # 3. Burn in, check, measure.
    key = jax.random.PRNGKey(seed)
    started = time.time()
    for _ in range(burn_in):
        key, subkey = jax.random.split(key)
        walkers, _ = move(params, walkers, subkey, width)
    log(f"burn-in of {burn_in} steps done after {time.time() - started:.0f} s (includes compiling)")

    training_energy = np.loadtxt(run / "train_stats.csv", delimiter=",", skiprows=1, usecols=1, ndmin=1)
    summary = dict(run=str(run), checkpoint=final.name, checkpoint_step=done,
                   checkpoint_md5=file_md5(final), network=settings["network"], eps=settings["eps"],
                   gradient=settings["gradient"], steps_requested=steps, burn_in=burn_in,
                   seed=seed, batch_size=batch, n_electrons=n_electrons,
                   mcmc_moves_per_step=train_moire.Settings().mcmc_steps, mcmc_width=float(width),
                   supercell=supercell.tolist(), density_bins=DENSITY_BINS,
                   r_max=float(r_max), radial_bins=RADIAL_BINS,
                   q_vectors=qs.tolist(), q_indices=q_indices.tolist())

    rows, totals = [], {}
    timer = time.time()
    for step in range(steps):
        key, subkey = jax.random.split(key)
        walkers, means, sums = evaluation_step(params, walkers, subkey)
        sums = jax.device_get(sums)
        add_to(totals, sums)
        # Per-step S(kappa) and site occupations: is the spatial pattern stationary?
        s_kappa = float(np.mean(sums["rho_q_squared"][at_kappa]) / (batch * n_electrons))
        occupations = {f"occupation_{k}": float(n) / batch for k, n in enumerate(sums["occupation"])}
        rows.append(dict(step=step, **{name: float(value) for name, value in means.items()},
                         S_kappa=s_kappa, **occupations))

        if step + 1 == check_steps:
            evaluated = float(np.mean([row["energy"] for row in rows]))
            trained = float(np.mean(training_energy[-CHECK_TRAINING_STEPS:]))
            summary["sanity_check"] = dict(evaluation_first_steps=evaluated,
                                           training_last_steps=trained, tolerance=CHECK_TOLERANCE)
            log(f"sanity check: evaluation {evaluated:.5f}, end of training {trained:.5f}")
            if not np.isfinite(evaluated) or abs(evaluated - trained) > CHECK_TOLERANCE:
                summary["sanity_check"]["passed"] = False
                summary.update(steps_done=step + 1)
                save(out, rows, totals, summary)
                sys.exit(f"sanity check FAILED: the first {check_steps} steps give {evaluated:.5f}, the "
                         f"end of training {trained:.5f}; not finite, or more than {CHECK_TOLERANCE} apart. "
                         "Either this is not the trained wave function, or the run had not converged.")
            summary["sanity_check"]["passed"] = True

        if (step + 1) % SAVE_EVERY == 0 or step + 1 == steps:
            elapsed = time.time() - timer
            summary.update(steps_done=step + 1, n_configurations=(step + 1) * batch,
                           seconds=elapsed, steps_per_second=(step + 1) / elapsed)
            save(out, rows, totals, summary)
            log(f"step {step + 1:6d}  mean energy {np.mean([r['energy'] for r in rows]):.5f}  "
                f"pmove {rows[-1]['pmove']:.3f}  {(step + 1) / elapsed:.2f} steps/s")
    return summary


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("run", type=Path, help="training folder, e.g. runs/stage08/training/attention_eps5")
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--burn-in", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run = args.run.resolve()
    out = (args.out or RUNS / "evaluation" / run.name).resolve()
    out.mkdir(parents=True, exist_ok=True)

    # Folder and GPU safety, before JAX is imported.
    folder_lock = train_moire.hold_lock(out / ".evaluation.lock",            # noqa: F841 -- held until exit
                                        f"another process is evaluating into {out}")
    settings, final = final_checkpoint(run)
    if not final.exists():
        sys.exit(f"{final} does not exist: has the run finished?")
    wanted = dict(checkpoint_md5=file_md5(final), steps_requested=args.steps,
                  burn_in=args.burn_in, seed=args.seed)
    existing = existing_evaluation(out, wanted)
    if existing == "finished" and not args.overwrite:
        print(f"{out} already holds this evaluation; use --overwrite to repeat it")
        return
    if existing == "other" and not args.overwrite:
        sys.exit(f"{out} holds a different or unfinished evaluation; use --overwrite to replace it")
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not gpu.isdigit():
        sys.exit("make exactly one GPU visible, e.g. CUDA_VISIBLE_DEVICES=0")
    train_moire.GPU_LOCKS.mkdir(parents=True, exist_ok=True)
    gpu_lock = train_moire.hold_lock(train_moire.GPU_LOCKS / f"gpu{gpu}.lock",  # noqa: F841 -- held until exit
                                     f"GPU {gpu} is in use by another job")

    import jax
    if jax.default_backend() != "gpu":
        sys.exit(f"JAX found no working GPU (backend: {jax.default_backend()}); not starting")
    jax.config.update("jax_default_matmul_precision", "float32")     # as the training
    evaluate(run, out, args.steps, args.burn_in, args.seed)
    print("finished")


if __name__ == "__main__":
    main()
