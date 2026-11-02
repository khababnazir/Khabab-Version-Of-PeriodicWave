# evaluation — our trained networks with their weights frozen

One folder per run, written by `learn/stage08_own_code/analysis/evaluate_weights.py`
(started for all four by `launch_evaluation.sh`), plus `logs/`. The final
weights of each run are fixed; only the 1024 walkers move, for 10,000 steps
after 200 steps of burn-in, with the Metropolis step width fixed at its final
training value. The same evaluation as for the reference run
(`runs/archive/stage08_authors_code/evaluation/`), with our code.

## What is in a run folder

| File | What it is |
|---|---|
| `energies.csv` | One row per evaluation step (10,000). Batch means over the 1024 walkers, in code units (× 3.375559 / 6 = meV per electron): `energy` and its parts `kinetic`, `moire`, `coulomb`; `variance`, the batch variance of the local energy; `pmove`, the fraction of moves accepted; `S_kappa`, the structure factor at κ for that step; `occupation_0` … `occupation_8`, how many electrons sit nearest each site, per walker. |
| `observables.npz` | Sums over all steps and walkers, from which the figures are made: `density` and `pairs` (90 × 90 histograms over the supercell), `pair_distances` (150 bins up to 1.5 a_M), `patterns` (how often each filling pattern of the 9 sites occurred), `occupation`, `occupation_products` (per site, and per pair of sites), `rho_q`, `rho_q_squared` (for S(q) at 126 wave vectors), and `steps_done`. |
| `summary.json` | Network, ε and gradient of the run, the checkpoint used and its md5 checksum, the settings, the sanity check against the end of training, timing. |
| `.evaluation.lock` | Keeps a second evaluation out of the folder while one runs. |

All three data files record how many steps they contain;
`plot_weights.py` refuses a folder where they disagree.

## logs/

`gpu0.log`: start, end and exit code of each evaluation and what it printed
(sanity check, progress every 1000 steps).

GPU records, every 10 seconds, both GPUs:

- `gpu_monitor_2026-10-29.csv`: the whole queue of 2026-10-29 (06:51–07:34):
  temperature, power, clock, fan, and why the clock is lowered (`0x20` =
  software thermal slowdown, `0x01` = idle). No utilization or memory
  columns yet.
- `gpu_usage_2026-10-29.csv`: started by hand at 07:22 during the same queue
  (its last two evaluations) and stopped after 50 minutes, with
  utilization and memory added.
- From now on `launch_evaluation.sh` writes one file per start,
  `gpu_monitor_<date>_<time>.csv`, with all of these columns.
