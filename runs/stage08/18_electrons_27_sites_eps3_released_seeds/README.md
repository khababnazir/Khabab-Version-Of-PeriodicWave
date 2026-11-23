# runs/stage08/18_electrons_27_sites_eps3_released_seeds — OUR code at eps = 3, released settings, two random starts

Asked for by Khabab on 2026-11-23, after the authors' code, with their released settings, found the
honeycomb Wigner crystal at eps = 3 from its seed 42 and not from its first, clock-seeded start
(`authors_exact_code_runs/runs/18_electrons_27_sites_eps3_seed42/`: frozen -96.1305(2) meV per
electron against -95.9339(2)). Does OUR code, with the same settings, find it too?

Two runs of `learn/stage08_own_code/train_moire.py` (unchanged), one per GPU, 18 electrons on 27
sites, eps = 3, the self-attention network, with the authors' RELEASED settings (`--settings
released`: 1024 walkers, 200,000 iterations, learning rate 0.1 / (1 + t/20000), 4 layers, 4 heads,
2 perceptrons, 4 determinants; 30 Metropolis steps per iteration), the authors' gradient
(`--gradient real`), folx + slogdet (needed at 18 electrons):

| GPU | seed | training | frozen evaluation |
|---|---|---|---|
| 0 | 42 | `training/attention_18e_eps3_seed42/` | `evaluation/attention_18e_eps3_seed42/` |
| 1 | 43 | `training/attention_18e_eps3_seed43/` | `evaluation/attention_18e_eps3_seed43/` |

Each run is followed, if it ends with exit code 0, by its frozen-weight evaluation
(`learn/stage08_own_code/analysis/evaluate_weights.py`, unchanged, `--steps 20000`: 20,000 steps x
1024 walkers = 20,480,000 configurations, as for the authors'-code runs).

**Seed 42 here is NOT the authors' seed 42.** A seed only names a stream of random numbers; our code
draws its initial weights, walkers and moves from that stream in its own order and with its own
initialisation, so our seed 42 gives a different start from theirs. These are two more random
starts, not a copy of the authors' successful one. No seed number is special: what the authors'
runs show is that the end state depends on the start.

Our earlier eps = 3 run (`../18_electrons_27_sites_eps3_real_gradient/`) used the paper's Table II
settings and seed 0 and ended liquid-like (frozen -95.8838(2)); these runs use the released
settings, as the authors'-code runs.

## How it was started

```bash
cd ~/moire-nnvmc/runs/stage08/18_electrons_27_sites_eps3_released_seeds
mkdir -p training/logs
setsid nohup bash how_this_was_started.sh 0 42 >> training/logs/gpu0.log 2>&1 < /dev/null &
setsid nohup bash how_this_was_started.sh 1 43 >> training/logs/gpu1.log 2>&1 < /dev/null &
```

Monitors every 10 s in `training/logs/` (GPU: nvidia-smi; CPU and memory: system_monitor.py).

## Log

- 2026-11-23, before the start: the launcher was dry-run with a fake python (the training command
  in learn/stage08_own_code with exactly the options above; the evaluation only after exit code 0;
  bad arguments refused with exit code 2).
- 2026-11-23 04:19:55: both runs started (2 GPUs). `settings.json` of each: batch 1024,
  200,000 iterations, learning rate 0.1, delay 20000, 4 layers, 4 heads, 2 perceptrons, mcmc 30
  steps, burn-in 300, gradient real, folx, slogdet, released network form; seeds 42 and 43.
  Speed after 2,000 iterations: 0.23 s per iteration on GPU 0, 0.25 to 0.30 on GPU 1 (its power
  limit is 250 W): about 13 and 14 to 17 hours, then about 1.3 hours of evaluation each. No NaN.
- 2026-11-23 05:36: check at about 18,000 iterations (both running, no NaN, no GPU error). Learn's
  checkpoint_structure.py (unchanged) on the checkpoints so far (`results/checkpoint_structure_*.csv`):
  NO honeycomb in either run. Seed 42: S(kappa) 0.600 at 7,254 and 0.603 at 14,853, sublattice shares
  0.334 / 0.333 / 0.333 and 0.324 / 0.325 / 0.351. Seed 43: 0.590 at 6,843 and 0.592 at 14,014,
  0.332 / 0.332 / 0.336 and 0.327 / 0.339 / 0.335. For comparison, the authors' code: seed 42 was
  already a honeycomb at its first checkpoint, 4,709 (S(kappa) 2.69, shares 0.484 / 0.484 / 0.032);
  their first, clock-seeded run was not (0.65 at 4,759; 0.306 / 0.306 / 0.388).
  Training energy, meV per electron, mean over iterations 16,500 to 17,499: ours -95.8464 (seed 42)
  and -95.8363 (seed 43); the authors' code -96.1061 (seed 42, honeycomb) and -95.8077 (first run).

## Results

After the runs.
