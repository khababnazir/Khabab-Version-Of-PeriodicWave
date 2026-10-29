# runs/stage08 — data of our own Stage 8

The four runs of the paper's Fig. 5 trained with OUR code
(`learn/stage08_own_code/`). The run with the authors' code, which these are
compared with, is in `runs/archive/stage08_authors_code/`.

| Folder | What is in it | Made by |
|---|---|---|
| `training/` | The four training runs: energy at every iteration, checkpoints, and the launcher logs. | `launch_fig5.sh` → `train_moire.py` |
| `evaluation/` | The four final wave functions with their weights frozen: energies, observables, logs. | `analysis/launch_evaluation.sh` → `analysis/evaluate_weights.py` |
| `results/` | The result tables, and ours against the reference run. | `analysis/plot_weights.py` |
| `figures/` | Figures 1–7. | `analysis/analyse_runs.py` (1), `analysis/plot_weights.py` (2–7) |

(`analysis/` means `learn/stage08_own_code/analysis/`.)

Each folder has its own `README.md`.
