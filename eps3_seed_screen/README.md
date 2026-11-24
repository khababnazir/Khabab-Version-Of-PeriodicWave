# eps3_seed_screen — how often does a random start find the honeycomb? (eps = 3, both codes)

Asked for by Khabab on 2026-11-23. At eps = 3 the authors' code (released settings, self-attention)
ended in the honeycomb Wigner crystal from its seed 42 and in the even, liquid-like state from its
first, clock-seeded start; the honeycomb is 0.197 meV per electron lower (frozen weights;
`authors_exact_code_runs/runs/18_electrons_27_sites_eps3_seed42/`). Our code with the same settings
from two random starts (seeds 42 and 43, `runs/stage08/18_electrons_27_sites_eps3_released_seeds/`)
had no honeycomb after about 15,000 iterations and was stopped. One or two starts cannot say how
often a start finds the crystal, in either code; this screen runs many short starts of each.

Every start: 18 electrons on 27 sites, eps = 3, the self-attention network, the authors' released
settings (1024 walkers, learning rate 0.1 / (1 + t/20000), 4 layers, 4 heads, 2 perceptrons,
4 determinants), 5,000 iterations. The learning rate depends only on the iteration, so these
5,000 iterations are exactly the start of a 200,000-iteration run. Why 5,000: the authors' seed-42
run was a honeycomb at its first checkpoint (4,709), and its training energy had separated from
the liquid-like run's by iterations 750 to 1,000.

| GPU | code | seeds | each start | all |
|---|---|---|---|---|
| 0 | the authors' (`authors/seed_S/`) | 1 to 12 | about 31 min | about 6.3 h |
| 1 | ours (`ours/seed_S/`) | 1 to 12 | about 25 min | about 5 h |

Seeds of the two codes with the same number are NOT the same start: each code draws its weights,
walkers and moves from the seed in its own way.

## The authors' code with any seed (changes.diff)

Their train.py takes its seed from the clock, or uses 42 with `cfg.debug.deterministic`; it has
no option for another seed. `run_2deg_triangular_potential_screen.py` is their script with
`cfg.debug.deterministic = True`, 5,000 iterations, and a seed override: their train.py then calls
`jax.random.PRNGKey(42)` once (train.py line 373, the only PRNGKey call in periodicwave), and the
copy turns that one call into `PRNGKey(SCREEN_SEED)` (an environment variable). Nothing of their
code is changed.

`check_seed_override.sh` (run on GPU 0 before the start, 2026-11-23 05:41 to 05:50; output in
`logs/check_seed_override.log`): three runs of 3 iterations each. All checks passed:
- SCREEN_SEED=42 starts exactly as the authors' seed-42 run (iteration 0: pmove 0.16832682, energy
  -312.667, both identical), and its config.json differs from that run's only in optim.iterations;
- SCREEN_SEED=1 starts elsewhere (pmove 0.16871746, energy -304.21844), with the same config.json;
- a mutant without the override line, SCREEN_SEED=1, starts as seed 42: the check catches it.

## How it was started

```bash
cd ~/moire-nnvmc/eps3_seed_screen
setsid nohup flock -n ~/moire-nnvmc/runs/gpu_locks/gpu0.lock bash how_this_was_started.sh 0 authors 1 12 >> logs/gpu0.log 2>&1 < /dev/null &
setsid nohup bash how_this_was_started.sh 1 ours 1 12 >> logs/gpu1.log 2>&1 < /dev/null &
```

(`train_moire.py` takes the GPU lock itself, so the queue of our code must not hold it.) The
launcher was dry-run with a fake python: the commands above for each seed, a finished authors' seed
skipped, a busy GPU (exit code 75) stops the queue, bad arguments refused.

## Our code from the authors' lucky start, and with the field (added 2026-11-23)

Khabab, after the first 9 starts (no honeycomb): try our network from the authors' lucky start, and
with the honeycomb field, judged early like the screen, and more seeds of our code.

- **Lucky start** (`lucky_start/seed_S/`, 5,000 iterations each, our seeds 1 to 4).
  `capture_authors_start.py` runs the authors' seed-42 script (unchanged) in a temporary folder
  and records what their train.py makes before any burn-in: the initial parameters (their
  network.init) and the 1024 initial walkers (their init_electrons_gaussian); then it stops the
  run. `lucky_start.py` converts the parameters to our network (a copy of
  learn/stage08_own_code/test_networks.py's converter) and writes them, with those walkers, as an
  iteration-0 checkpoint for our train_moire.py, which then runs its usual burn-in and training
  (it continues from any checkpoint, and at iteration 0 it burns in first). The seed S of our
  run only chooses our Metropolis moves and KFAC's random numbers: those cannot be the authors'.
  So this is the authors' starting POINT, not their path.
  The start used is captured on the GPU that runs these starts (`authors_seed42_start_gpu.pkl`,
  as the authors' run was on a GPU; its difference from the CPU capture
  `authors_seed42_start.pkl` is printed in the log).
- **Field** (`field/seed_42/`): our code from seed 42 (the start of our stopped seed-42 run, which
  stayed liquid-like) with the field of the authors'-code test (lambda = 0.5: sublattice 0 raised
  by 7.6 meV) for iterations 0 to 4,999 (`field_training.py` wraps train_moire.make_problem so
  that the local energy includes the field), then the field off to 14,999 (plain train_moire.py,
  same folder). `field.json` in the folder records the field.
- **More seeds of our code**: `ours/seed_13` to `seed_24`, as the first 12.

`test_lucky_and_field.py` (CPU), 7 tests, all pass: the capture gives the same start twice, bit
for bit, and its walkers are the authors' Gaussian start; the converter copy is learn's function;
our network with the converted parameters gives their network's log|Psi| and phase (16
configurations spread over the cell); the iteration-0 checkpoint reads back with train_moire's
own loader (iteration 0, these parameters, these walkers in their layout, width 0.2, the key a
fresh start with that seed has); the field is 3 lambda on sublattice 0 and -1.5 lambda elsewhere,
periodic, and the wrapped local energy is the plain one plus exactly the field. Three mutants,
each caught: queries and keys swapped in the converter, the walkers read in the wrong layout, the
field term dropped. `start_lucky_and_field.sh` was dry-run with a fake python.

Queued on 2026-11-23 08:22 to follow the screen on each GPU:
- GPU 1, when the screen of our code ends (about 10:45): `bash start_lucky_and_field.sh 1 lucky 1 4`,
  then `bash start_lucky_and_field.sh 1 field 42` (about 1.6 h, then about 1.2 h);
- GPU 0, when the screen of the authors' code ends (about 12:10): `bash how_this_was_started.sh 0 ours 13 24`
  (about 5 h).
Logs: `logs/gpu1_lucky_and_field.log`, `logs/gpu0.log`.

Overnight extension (Khabab, 2026-11-23 08:27 server time: run until he is back next morning,
about 23:30 server time), queued after the runs above on each GPU:
- GPU 1: `bash start_lucky_and_field.sh 1 lucky 5 8` (four more lucky starts), then
  `bash how_this_was_started.sh 1 ours 25 36` (until about 20:00);
- GPU 0: `flock -n ~/moire-nnvmc/runs/gpu_locks/gpu0.lock bash how_this_was_started.sh 0 authors 13 20`
  (eight more starts of the authors' code, until about 21:30).

## Long runs (2026-11-24)

Khabab: continue the start in which OUR code found the honeycomb without the field (random seed
36), and train SlaterNet from the same seed number on the other GPU. Both to 200,000 iterations,
then the frozen-weight evaluation (20,000 steps x 1024 walkers), into `long/` (the screen folders
stay as they were):
- GPU 0: `bash continue_to_200000.sh 0 ours/seed_36 ours_seed_36` -- a copy of ours/seed_36 in
  long/ours_seed_36/, continued by train_moire.py from its checkpoint 5,000 (the first 5,000 rows
  of train_stats.csv checked unchanged). Started 2026-11-24 00:55:57; 0.23 s per iteration, about
  12.5 h, then about 1.3 h of evaluation.
- GPU 1: `bash train_to_200000.sh 1 slaternet 36` -- a fresh SlaterNet run, long/slaternet_seed_36/
  (released SlaterNet sizes: 1 determinant, 4 layers, width 64, 2 perceptrons, as the authors'
  SlaterNet config). Seed 36 of SlaterNet is a different start from seed 36 of the attention
  network (another network, other initial weights). Started 00:57:38; 0.076 s per iteration,
  about 4.2 h, then its evaluation.
Both launchers were dry-run with a fake python. Logs: `logs/long_ours_seed_36.log`,
`logs/long_slaternet_seed_36.log`.

## Results

`screen_results.py` -> `results/screen.csv`: for every finished start, S(kappa) and the sublattice
shares of the walkers at iteration 5,000, the training energy over iterations 4,000 to 4,999, and
whether the emptiest sublattice holds less than 15 % (honeycomb). Started 2026-11-23 05:51:33;
results after the screen.
