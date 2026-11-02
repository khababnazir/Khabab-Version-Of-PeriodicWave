# 18 electrons, 27 sites — evaluation: our trained networks with their weights frozen

One folder per evaluation, written by `learn/stage08_own_code/analysis/evaluate_weights.py`
(started by `launch_evaluation.sh GPU 18_electrons_27_sites ...`), plus `logs/`. The weights
are fixed; only the 4096 walkers move: 200 steps of burn-in, then 5,000 steps of 30 moves,
with the step width fixed at its final training value. 5,000 × 4,096 = 20,480,000 sampled
configurations per evaluation.

| Folder | Weights evaluated |
|---|---|
| `attention_18e_eps5`, `attention_18e_eps10`, `slaternet_18e_eps5`, `slaternet_18e_eps10` | the final checkpoint (150,000 iterations) of the run of the same name in `../training/` |
| `slaternet_18e_eps10_at_033545` | `slaternet_18e_eps10` after 33,545 iterations, near its lowest training energy |

## In each folder

| File | What it is |
|---|---|
| `energies.csv` | One row per evaluation step (5,000). Batch means over the 4096 walkers for the **whole system**, in code units — **× 3.375559 / 18 = meV per electron**: `energy` and its parts `coulomb`, `kinetic`, `moire`; `variance`; `pmove`; `S_kappa`, the structure factor at κ for that step; `occupation_0` … `occupation_26`, electrons nearest each of the 27 sites, per walker. |
| `observables.npz` | Counts summed over all steps: `density` and `pairs` (156 × 156 bins over the cell), `pair_distances` (150 bins up to 2.6 a_M), `occupation` (27), `occupation_products` (27 × 27), `rho_q`, `rho_q_squared` (384 wave vectors), `steps_done`. |
| `summary.json` | `n_electrons` 18, `n_sites` 27, the checkpoint and its md5, the training settings, the sanity check, the timing. Its `run` path is where the run was on 2026-11-02 (`runs/stage08/table1/...`, before the folders were renamed). |
