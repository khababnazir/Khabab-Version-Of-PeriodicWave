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

## Results

`screen_results.py` -> `results/screen.csv`: for every finished start, S(kappa) and the sublattice
shares of the walkers at iteration 5,000, the training energy over iterations 4,000 to 4,999, and
whether the emptiest sublattice holds less than 15 % (honeycomb). Started 2026-11-23 05:51:33;
results after the screen.
