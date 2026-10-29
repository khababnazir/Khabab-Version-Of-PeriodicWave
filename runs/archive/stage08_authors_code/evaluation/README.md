# evaluation — the trained networks with their weights frozen

One folder per run, written by `learn/archive/stage08_authors_code/analysis/evaluate_weights.py`
(started for all four by `launch_evaluation.sh`), plus `logs/`. The final
weights of each run are fixed; only the 1024 walkers move, for 10,000 steps
after 200 steps of burn-in. Each run folder has a short `README.md` with its
own numbers.

## What is in a run folder

| File | What it is |
|---|---|
| `energies.csv` | One row per evaluation step (10,000). Batch means over the 1024 walkers, in code units (× 3.375559 / 6 = meV per electron): `energy` and its parts `kinetic`, `moire`, `coulomb`; `variance`, the batch variance of the local energy; `pmove`, the fraction of moves accepted; `S_kappa`, the structure factor at κ for that step; `occupation_0` … `occupation_8`, how many electrons sit nearest each site, per walker. |
| `observables.npz` | Sums over all steps and walkers, from which the figures are made: `density` and `pairs` (90 × 90 histograms over the supercell), `pair_distances` (150 bins up to 1.5 a_M), `patterns` (how often each filling pattern of the 9 sites occurred), `occupation`, `occupation_products` (per site, and per pair of sites), `rho_q`, `rho_q_squared` (for S(q) at 126 wave vectors), and `steps_done`. |
| `summary.json` | Settings, the checkpoint used and its md5 checksum, the sanity check against the end of training, timing. Its `run` entry is the training folder's path at the time, before the move to `training/`. |
| `.evaluation.lock` | Keeps a second evaluation out of the folder while one runs. |

All three data files record how many steps they contain;
`plot_weights.py` refuses a folder where they disagree.

## logs/

`gpu0.log` and `gpu1.log`: start, end and exit code of each evaluation and
what it printed (sanity check, progress every 1000 steps). Later starts that
found an evaluation already finished say so and leave it untouched.
