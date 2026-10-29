# runs/archive/stage08_authors_code — data of the archived Stage 8

**Archived on 2026-10-28.** These are the four runs made with the authors' own
code. They are kept, unchanged, as the reference for our own Stage 8
(`learn/stage08_own_code/`, data in `runs/stage08/`).

Everything Stage 8 produced. The code that made it is in
`learn/archive/stage08_authors_code/` (training) and `learn/archive/stage08_authors_code/analysis/`
(everything after training). This folder is not in git (it is large and can
be regenerated), so these notes live only here.

| Folder | What is in it | Made by |
|---|---|---|
| `training/` | The four training runs of the paper's Fig. 5 — the energy at every iteration and the network weights (checkpoints) — and the launcher logs. **The raw results of training.** | `launch_fig5.sh` → `run_moire.py` |
| `evaluation/` | The four trained networks evaluated with frozen weights: energies step by step, and the histograms behind the figures. | `analysis/launch_evaluation.sh` → `evaluate_weights.py` |
| `results/` | **The main numbers**, two small tables: `energies_meV_per_electron.csv` and `electron_structure.csv`. Start here. | `analysis/plot_weights.py` |
| `figures/` | Six figures, numbered in reading order, and a note on each. | `analysis/analyse_runs.py`, `analysis/plot_weights.py` |

Each folder has its own `README.md`.

## The four runs

| Run | Network | ε | Training (UTC, 26 Oct 2026) |
|---|---|---|---|
| `psiformer_eps10` | self-attention | 10 | 02:14 – 13:46, GPU 1 |
| `slaternet_eps10` | SlaterNet | 10 | 13:46 – 22:56, GPU 1 |
| `psiformer_eps5` | self-attention | 5 | 02:14 – 13:45, GPU 0 |
| `slaternet_eps5` | SlaterNet | 5 | 13:45 – 22:56, GPU 0 |

"psiformer" is the authors' name for their self-attention network
(`CustomPsiformer`). All four: 6 electrons, 9 sites, the authors' released
configuration, 200,000 iterations, batch 1024.

Moved twice. On 2026-10-27 the runs went from `runs/stage08/<run>` to
`runs/stage08/training/<run>`; on 2026-10-28 the whole folder went from
`runs/stage08/` to `runs/archive/stage08_authors_code/`. Files written
before then (`config.json`, `summary.json`, logs) still show the old paths.
The GPU locks moved to `runs/gpu_locks/`, shared by every stage.
