# runs/stage08 — data of our own Stage 8

Two systems, trained with our code (`learn/stage08_own_code/`), one folder each:

| Folder | System | Paper | Analysis scripts |
|---|---|---|---|
| `6_electrons_9_sites/` | 6 electrons on 9 sites | Fig. 5 | `learn/stage08_own_code/analysis/6_electrons_9_sites/` |
| `18_electrons_27_sites/` | 18 electrons on 27 sites | Table I, Fig. 6 | `learn/stage08_own_code/analysis/18_electrons_27_sites/` |

Each holds `training/` (the runs: settings, energy of every iteration, checkpoints = the
model weights, launcher logs), `evaluation/` (the final weights evaluated frozen),
`results/` (tables) and `figures/`, and a `README.md` with the results.
Each run folder is named `<network>_<electrons>e_eps<ε>`, e.g. `slaternet_18e_eps5` is
SlaterNet with 18 electrons at ε = 5 (until 2026-11-02 `slaternet_eps5`; logs and
`summary.json` files written before still show the old names).
The frozen-weight evaluation itself (`analysis/evaluate_weights.py`, `analysis/observables.py`)
is shared by both systems.

The earlier run with the authors' code (6 electrons) is in `runs/archive/stage08_authors_code/`.
