# runs/stage08/6_electrons_9_sites — the paper's Fig. 5, our own code

The four runs of the paper's Fig. 5 (6 electrons on 9 sites) trained with OUR code
(`learn/stage08_own_code/`). The run with the authors' code, which these are
compared with, is in `runs/archive/stage08_authors_code/`.

| Folder | What is in it | Made by |
|---|---|---|
| `training/` | The four training runs: energy at every iteration, checkpoints (the model weights), and the launcher logs. | `launch_fig5.sh` → `train_moire.py` |
| `evaluation/` | The four final wave functions with their weights frozen: energies, observables, logs. | `analysis/launch_evaluation.sh 0 6_electrons_9_sites` → `analysis/evaluate_weights.py` |
| `results/` | The result tables, and ours against the reference run. | `analysis/6_electrons_9_sites/plot_weights.py` |
| `figures/` | Figures 1–7. | `analysis/6_electrons_9_sites/analyse_runs.py` (1), `analysis/6_electrons_9_sites/plot_weights.py` (2–7) |

(`analysis/` means `learn/stage08_own_code/analysis/`.)

Each folder has its own `README.md`. Until 2026-11-02 these four folders were directly in
`runs/stage08/`; the `run` paths inside the evaluations' `summary.json` files still show
where they were then.
