# runs/stage08 — data of our own Stage 8

Two systems, trained with our code (`learn/stage08_own_code/`), one folder each:

| Folder | System | Paper | Analysis scripts |
|---|---|---|---|
| `6_electrons_9_sites/` | 6 electrons on 9 sites | Fig. 5 | `learn/stage08_own_code/analysis/6_electrons_9_sites/` |
| `18_electrons_27_sites/` | 18 electrons on 27 sites | Table I, Fig. 6 | `learn/stage08_own_code/analysis/18_electrons_27_sites/` |

Each holds `training/` (the runs: settings, energy of every iteration, checkpoints = the
model weights, launcher logs), `evaluation/` (the final weights evaluated frozen),
`results/` (tables) and `figures/`, and a `README.md` with the results.
The frozen-weight evaluation itself (`analysis/evaluate_weights.py`, `analysis/observables.py`)
is shared by both systems.

The earlier run with the authors' code (6 electrons) is in `runs/archive/stage08_authors_code/`.
