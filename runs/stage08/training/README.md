# training — the four runs, with our code

| Run | Network | ε | GPU |
|---|---|---|---|
| `slaternet_eps10` | SlaterNet | 10 | 0, first |
| `attention_eps10` | self-attention | 10 | 0, second |
| `slaternet_eps5` | SlaterNet | 5 | 1, first |
| `attention_eps5` | self-attention | 5 | 1, second |

All four: 6 electrons, 9 sites, the reference run's settings, 200,000
iterations, batch 1024, the authors' gradient convention (`--gradient real`).

In each run folder (see the docstring of `train_moire.py`):

| File | What it is |
|---|---|
| `settings.json` | what the run is; checked on every restart |
| `train_stats.csv` | one row per iteration: `step`, `energy` (mean Re E_loc, whole system, code units of 3.375559 meV), `variance`, `energy_imag` (mean Im E_loc, should be near 0), `pmove` (accepted Metropolis moves), `step_width`, `seconds` since the start |
| `checkpoint_NNNNNN.npz` | everything needed to continue after NNNNNN iterations; every 30 minutes and at the end |
| `starts.log` | one line per start, with the JAX and kfac-jax versions |

`logs/gpu0.log`, `logs/gpu1.log`: what each queue printed.
