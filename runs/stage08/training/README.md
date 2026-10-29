# training — the four runs, with our code

| Run | Network | ε | GPU |
|---|---|---|---|
| `slaternet_eps10` | SlaterNet | 10 | 0, first |
| `attention_eps10` | self-attention | 10 | 0, second |
| `slaternet_eps5` | SlaterNet | 5 | 1, first |
| `attention_eps5` | self-attention | 5 | 1, second; finished on 0 (see below) |

All four: 6 electrons, 9 sites, the reference run's settings, 200,000
iterations, batch 1024, the authors' gradient convention (`--gradient real`).

In each run folder (see the docstring of `train_moire.py`):

| File | What it is |
|---|---|
| `settings.json` | what the run is; checked on every restart |
| `train_stats.csv` | one row per iteration: `step`, `energy` (mean Re E_loc, whole system, code units of 3.375559 meV), `variance`, `energy_imag` (mean Im E_loc, should be near 0), `pmove` (accepted Metropolis moves), `step_width`, `seconds` since the start |
| `checkpoint_NNNNNN.npz` | everything needed to continue after NNNNNN iterations; every 30 minutes and at the end |
| `starts.log` | one line per start, with the JAX and kfac-jax versions |

`logs/`: what each queue printed, and a GPU temperature record (see `logs/README.md`).

## Results

Mean energy over the last 20% of training (iterations 160,000–199,999), in
meV per electron, with its blocking error in brackets.  The reference runs
(`runs/archive/stage08_authors_code/training/`) are averaged the same way.

| Run | Ours | Reference | Ours − reference | Fig. 5 (read by eye) |
|---|---|---|---|---|
| `slaternet_eps10` | −31.6258(6) | −31.5948(16) | −0.0310(17) | −31.6 |
| `slaternet_eps5` | −57.9120(9) | −57.9244(10) | +0.0123(14) | −57.9 |
| `attention_eps10` | −32.2775(1) | −32.2759(1) | −0.0016(1) | −32.3 |
| `attention_eps5` | −59.5328(1) | −59.5332(1) | +0.0004(1) | −59.5 |

These are training averages: the weights still change from one iteration to
the next.  The evaluation with frozen weights is still to do.

## The interruption of `attention_eps5`

- 2026-10-28 16:00:16, iteration 190,863: GPU 1 dropped off the PCIe bus
  (kernel log: Xid 79) and the run stopped.  The cause is not established.
  At the time the server's second power supply had no mains cable (connected
  2026-10-29), and GPU 0 is known to run close to its temperature limit
  (`logs/README.md`).
- 2026-10-29 00:17: a restart found no working GPU, ran on the CPU and was
  stopped by hand.  `train_stats.csv` was cut back to 174,540 rows, the
  newest checkpoint; iterations 174,540–190,862 of the first attempt are not
  kept.
- 2026-10-29 05:41: after a reboot (NVIDIA driver 595.99.02 instead of
  595.91.07), the run continued from `checkpoint_174540.npz` on GPU 0 and
  finished at 06:24:58.
- The restart left no visible mark.  Mean energy before and after iteration
  174,540 (160,000–174,539 against 174,540–199,999): ours −59.5325(1) and
  −59.5329(1); the reference run, split at the same iteration and never
  interrupted, −59.5328(2) and −59.5334(1).  Both fall by about the same
  amount, which is the slow improvement of training, not the restart.
