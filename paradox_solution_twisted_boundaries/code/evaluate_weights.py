"""
PARADOX EXPERIMENT -- twisted boundary conditions.  THIS FILE IS A COPY.

It is learn/stage08_own_code/analysis/evaluate_weights.py (repository commit
b6f773f).  The only change: it imports the train_moire.py of THIS folder (the
copy that knows --twist) instead of the one in learn/, so the evaluated wave
function carries the twist stored in the run's settings.json.  The original
refuses twisted runs ("settings this code does not know").  observables.py is
not copied: it is read, unchanged, from learn/.  ../CHANGES.diff lists every
changed line.  Run it from this folder (launch_twisted.sh does); the text from
"Stage 8" on is the original's.

Stage 8 (our code) -- evaluate a trained network with its weights FROZEN.

Run (one GPU, one evaluation), from learn/stage08_own_code/analysis:
    CUDA_VISIBLE_DEVICES=0 ../../../.venv/bin/python evaluate_weights.py ../../../runs/stage08/6_electrons_9_sites_real_gradient/training/attention_6e_eps5
Options: --steps N       evaluation steps after burn-in (default 10000 at 6 electrons,
                         5000 at 18: with 4096 walkers that is 20,480,000 configurations)
         --burn-in N     steps that only move the walkers, first (default 200)
         --seed S        (default 0)
         --checkpoint N  evaluate checkpoint_N.npz instead of the final one
         --out DIR       default: the evaluation/ folder beside the run's training/
                         folder, e.g. runs/stage08/18_electrons_27_sites_real_gradient/evaluation/<run name>;
                         with --checkpoint, <run name>_at_<N>
         --overwrite     replace an evaluation that is already in DIR

18 electrons on 27 sites (added 2026-11-02, for Table I and Fig. 6).  The
network, its sizes and the local energy (folx + slogdet) come from the run's
own settings.json, exactly as the training built them; the supercell is the
27-site one.  The sanity-check tolerance grows with the number of electrons
(the same 0.028 meV per electron), and the density and pair maps keep 30
bins per moire length (90 x 90 in the 9-site cell, 156 x 156 in the 27-site
cell).  Filling patterns are counted in the 9-site cell only
(observables.py says why).

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
evaluation steps must be finite and within 0.05 code units per 6 electrons
(~0.03 meV per electron; 0.15 at 18 electrons) of the mean of the 2000
training iterations just before the checkpoint.  A wrong network, wrong
weights or a wrong Hamiltonian would typically miss by far more.  It is a
sanity check, not a proof: different wave functions can have similar
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
                      S(kappa) and the 9 (or 27) site occupations of that step
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
# CHANGED IN THIS COPY (the two path lines only).
sys.path.insert(0, str(HERE.parents[1] / "learn" / "stage08_own_code" / "analysis"))   # observables.py, unchanged
sys.path.insert(0, str(HERE))             # train_moire.py of THIS folder: the copy with the twist
import train_moire                        # noqa: E402 -- imports JAX only inside its functions

BINS_PER_MOIRE_LENGTH = 30     # density and pair maps: 90 x 90 bins (9 sites), 156 x 156 (27 sites)
RADIAL_BINS = 150
CHECK_STEPS = 200              # evaluation steps used in the sanity check
CHECK_TRAINING_STEPS = 2000    # training steps just before the checkpoint it is compared with
CHECK_TOLERANCE = 0.05 / 6     # code units of energy per electron: 0.05 for 6 electrons, 0.15 for 18
DEFAULT_STEPS = {6: 10000, 18: 5000}
SAVE_EVERY = 1000
KAPPA = 4 * np.pi / 3          # corner of the moire Brillouin zone (a_M = 1)
G = 4 * np.pi / np.sqrt(3)     # |moire reciprocal vector|


# ------------------------------------------------- the trained wave function

def run_settings(run):
    """The run's settings.json, with the entries that folders made before 2026-10-30 lack
    filled in exactly as train_moire.py does, and its training Settings."""
    import dataclasses
    saved = json.loads((run / "settings.json").read_text())
    saved.setdefault("electrons", 6)
    saved["settings"] = {**train_moire.BEFORE_18_ELECTRONS, **saved.get("settings", {})}
    known = {field.name for field in dataclasses.fields(train_moire.Settings)}
    unknown = sorted(set(saved["settings"]) - known)
    if unknown:
        sys.exit(f"{run / 'settings.json'} has settings this code does not know: {unknown}")
    return saved, dataclasses.replace(train_moire.Settings(), **saved["settings"])


def checkpoint_path(run, saved, step=None):
    """checkpoint_<step>.npz of the run; the final one if step is None."""
    return run / f"checkpoint_{saved['iterations'] if step is None else step:06d}.npz"


def default_out(run, saved, step=None):
    """Where an evaluation goes unless --out says otherwise: the evaluation/ folder beside
    the training/ folder that holds the run, e.g.
        runs/stage08/18_electrons_27_sites_real_gradient/training/attention_18e_eps5
     -> runs/stage08/18_electrons_27_sites_real_gradient/evaluation/attention_18e_eps5
    """
    if run.parent.name != "training":
        sys.exit(f"{run} is not inside a training/ folder; say where the evaluation goes with --out")
    root = run.parent.parent / "evaluation"
    if step is None or step == saved["iterations"]:
        return root / run.name
    return root / f"{run.name}_at_{step:06d}"


def training_window(training_stats, done):
    """The training energies of the CHECK_TRAINING_STEPS iterations just before checkpoint `done`.

    training_stats: columns step and energy of train_stats.csv.  checkpoint_<done>
    holds the weights after iterations 0 ... done-1, so these are the rows with
    step done-2000 ... done-1 (chosen by their step, not by their position).
    A run restarted from a checkpoint repeats the steps after it; the LAST row
    of a step is the one that led to the later weights, so that one is used.
    """
    reversed_steps = training_stats[::-1, 0]
    steps, first_in_reversed = np.unique(reversed_steps, return_index=True)
    energy = training_stats[::-1, 1][first_in_reversed]          # the last row of every step
    window = (steps >= done - CHECK_TRAINING_STEPS) & (steps < done)
    if window.sum() != min(done, CHECK_TRAINING_STEPS):
        sys.exit(f"train_stats.csv does not hold all {CHECK_TRAINING_STEPS} iterations before {done}")
    return energy[window]


def potential_parts(eps, electrons=6):
    """parts(R) -> (moire, Coulomb) energy of one configuration R of shape (N, 2).

    The same functions and constants as the local energy of
    train_moire.make_problem, so local energy - moire - Coulomb is exactly
    the kinetic energy (test_evaluate_weights.py checks it).
    """
    import slaternet
    from local_energy import moire_potential
    from ewald import make_ewald_setup, coulomb_energy

    V, U, phi = train_moire.moire_constants(eps)
    ewald = make_ewald_setup(slaternet.supercell_vectors(train_moire.SITES[electrons]))

    def parts(R):
        return moire_potential(R, V, phi), coulomb_energy(ewald, R, strength=U)
    return parts


def file_md5(path):
    with open(path, "rb") as handle:
        return hashlib.md5(handle.read()).hexdigest()


# ------------------------------------------------------------------- output

OBSERVABLE_ARRAYS = ("density", "pairs", "pair_distances", "occupation", "occupation_products",
                     "rho_q", "rho_q_squared")    # the sums observables.measure returns; at 9 sites also "patterns"


def existing_evaluation(folder, wanted):
    """None if the folder holds no evaluation, 'finished' if it holds this one, complete, else 'other'.

    `wanted` holds what identifies an evaluation: the checkpoint's md5
    checksum, the steps requested, the burn-in and the seed.  Complete means
    all of these:
      - the three files hold the requested number of steps (an overwrite
        stopped between two file writes leaves them disagreeing);
      - observables.npz holds every observable (at 9 sites also the patterns);
      - every number in energies.csv and observables.npz is finite;
      - the sanity check passed.  It runs after CHECK_STEPS steps, so only a
        shorter evaluation (in a test) may have none.
    """
    summary_file = folder / "summary.json"
    if not summary_file.exists():
        return None
    try:                       # a missing, damaged or half-written file counts as unfinished
        summary = json.loads(summary_file.read_text())
        with open(folder / "energies.csv", newline="") as handle:
            rows = list(csv.DictReader(handle))                       # one row per step
        finite_rows = all(np.isfinite(float(value)) for row in rows for value in row.values())
        with np.load(folder / "observables.npz") as observables_file:
            arrays = {name: observables_file[name] for name in observables_file.files}
    except Exception:
        return "other"
    sites = summary.get("n_sites") or train_moire.SITES.get(summary.get("n_electrons"))
    needed = ("steps_done",) + OBSERVABLE_ARRAYS + (("patterns",) if sites == 9 else ())
    every_array = all(name in arrays for name in needed)
    finite_arrays = all(np.isfinite(a).all() for a in arrays.values() if np.issubdtype(a.dtype, np.inexact))
    check = summary.get("sanity_check")
    checked = (check.get("passed") is True) if check is not None else (summary.get("steps_done", 0) < CHECK_STEPS)
    same = all(summary.get(name) == value for name, value in wanted.items())
    complete = every_array and (len(rows) == int(arrays["steps_done"]) == summary.get("steps_done")
                                == summary.get("steps_requested"))
    return "finished" if same and complete and finite_rows and finite_arrays and checked else "other"


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

def evaluate(run, out, steps, burn_in, seed, checkpoint=None, check_steps=CHECK_STEPS, log=print):
    """Evaluate the weights of the training folder `run` into `out`.  Returns the summary.

    checkpoint: the iteration count of the checkpoint to evaluate; None = the final one.
    """
    import jax
    import jax.numpy as jnp
    import observables
    import slaternet

    # 1. The network, local energy and Metropolis moves of the training, and its weights.
    saved, training_settings = run_settings(run)
    electrons = saved["electrons"]
    final = checkpoint_path(run, saved, checkpoint)
    expected = saved["iterations"] if checkpoint is None else checkpoint
    problem = train_moire.make_problem(saved["network"], saved["eps"], electrons, training_settings)
    move, _ = train_moire.make_sampler(problem, training_settings, saved["gradient"])
    params_like = problem.init(jax.random.PRNGKey(0), problem.arch)
    done, params, walkers, width = train_moire.load_parameters(final, params_like)
    if done != expected:
        sys.exit(f"{final.name} holds {done} iterations, not {expected}")
    batch, n_electrons = walkers.shape[0], walkers.shape[1]
    if n_electrons != electrons:
        sys.exit(f"{final.name} holds walkers of {n_electrons} electrons, settings.json says {electrons}")

    # 2. What is measured.
    supercell = np.asarray(slaternet.supercell_vectors(train_moire.SITES[electrons]))  # columns L1, L2
    qs, q_indices = observables.supercell_q_vectors(supercell, q_max=2 * G)
    at_kappa = np.isclose(np.linalg.norm(qs, axis=1), KAPPA, atol=1e-6)
    r_max = 0.5 * np.linalg.norm(supercell[:, 0])                 # g(r) is valid up to half the cell size
    density_bins = int(round(BINS_PER_MOIRE_LENGTH * np.linalg.norm(supercell[:, 0])))
    tolerance = CHECK_TOLERANCE * n_electrons
    parts = potential_parts(saved["eps"], electrons)

    @jax.jit
    def evaluation_step(params, walkers, key):
        walkers, pmove = move(params, walkers, key, width)        # width: fixed, the final training value
        energy = jnp.real(jax.vmap(lambda R: problem.local_energy(params, R))(walkers))
        moire, coulomb = jax.vmap(parts)(walkers)
        means = dict(energy=energy.mean(), variance=energy.var(),
                     kinetic=(energy - moire - coulomb).mean(),
                     moire=moire.mean(), coulomb=coulomb.mean(), pmove=pmove)
        return walkers, means, observables.measure(walkers, supercell, qs, density_bins, r_max, RADIAL_BINS)

    # 3. Burn in, check, measure.
    key = jax.random.PRNGKey(seed)
    started = time.time()
    for _ in range(burn_in):
        key, subkey = jax.random.split(key)
        walkers, _ = move(params, walkers, subkey, width)
    log(f"burn-in of {burn_in} steps done after {time.time() - started:.0f} s (includes compiling)")

    training_stats = np.loadtxt(run / "train_stats.csv", delimiter=",", skiprows=1, usecols=(0, 1), ndmin=2)
    summary = dict(run=str(run), checkpoint=final.name, checkpoint_step=done,
                   final_checkpoint=done == saved["iterations"],
                   checkpoint_md5=file_md5(final), network=saved["network"], eps=saved["eps"],
                   gradient=saved["gradient"], steps_requested=steps, burn_in=burn_in,
                   seed=seed, batch_size=batch, n_electrons=n_electrons,
                   n_sites=train_moire.SITES[electrons], training_settings=saved["settings"],
                   mcmc_moves_per_step=training_settings.mcmc_steps, mcmc_width=float(width),
                   supercell=supercell.tolist(), density_bins=density_bins,
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
            trained = float(np.mean(training_window(training_stats, done)))
            summary["sanity_check"] = dict(evaluation_first_steps=evaluated,
                                           training_last_steps=trained, tolerance=tolerance)
            log(f"sanity check: evaluation {evaluated:.5f}, training before the checkpoint {trained:.5f}")
            if not np.isfinite(evaluated) or abs(evaluated - trained) > tolerance:
                summary["sanity_check"]["passed"] = False
                summary.update(steps_done=step + 1)
                save(out, rows, totals, summary)
                sys.exit(f"sanity check FAILED: the first {check_steps} steps give {evaluated:.5f}, the "
                         f"training before the checkpoint {trained:.5f}; not finite, or more than "
                         f"{tolerance:.3f} apart. "
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
    parser.add_argument("run", type=Path, help="training folder, e.g. runs/stage08/6_electrons_9_sites_real_gradient/training/attention_6e_eps5")
    parser.add_argument("--steps", type=int, default=None, help="default 10000 (6 electrons), 5000 (18)")
    parser.add_argument("--burn-in", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", type=int, default=None, help="default: the final checkpoint")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run = args.run.resolve()
    saved, _ = run_settings(run)
    steps = DEFAULT_STEPS[saved["electrons"]] if args.steps is None else args.steps
    out = (args.out or default_out(run, saved, args.checkpoint)).resolve()
    out.mkdir(parents=True, exist_ok=True)

    # Folder and GPU safety, before JAX is imported.
    folder_lock = train_moire.hold_lock(out / ".evaluation.lock",            # noqa: F841 -- held until exit
                                        f"another process is evaluating into {out}")
    final = checkpoint_path(run, saved, args.checkpoint)
    if not final.exists():
        sys.exit(f"{final} does not exist" + ("" if args.checkpoint else ": has the run finished?"))
    wanted = dict(checkpoint_md5=file_md5(final), steps_requested=steps,
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
    evaluate(run, out, steps, args.burn_in, args.seed, checkpoint=args.checkpoint)
    print("finished")


if __name__ == "__main__":
    main()
