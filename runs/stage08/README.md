# runs/stage08 — data of our own Stage 8

Two systems, trained with our code (`learn/stage08_own_code/`), one folder per system and
gradient convention:

- `_real_gradient`: the authors' convention, as in their code. Their local energy keeps only
  its real part (`external/PeriodicWave/periodicwave/pbc/hamiltonian.py`, line 234), so the
  gradient term 2 < Im d  d phase/d theta > is left out. This is the reproduction of the paper.
- `_full_gradient`: the same runs with that term kept (`train_moire.py --gradient full`),
  everything else identical. Our addition, run 2026-11-03 to 11-05; results in its README.

| Folder | System | Gradient | Paper | Analysis scripts |
|---|---|---|---|---|
| `6_electrons_9_sites_real_gradient/` | 6 electrons on 9 sites | authors' (real) | Fig. 5 | `learn/stage08_own_code/analysis/6_electrons_9_sites/` |
| `18_electrons_27_sites_real_gradient/` | 18 electrons on 27 sites | authors' (real) | Table I, Fig. 6 | `learn/stage08_own_code/analysis/18_electrons_27_sites/` |
| `18_electrons_27_sites_full_gradient/` | 18 electrons on 27 sites | full | Table I, Fig. 6 | the same, with `--gradient full` |

Until 2026-11-03 the two `_real_gradient` folders were named `6_electrons_9_sites/` and
`18_electrons_27_sites/`; `summary.json` and `branch.json` files written before still show the
old paths. The analysis CODE folders keep the system names: they serve both gradients.

Each holds `training/` (the runs: settings, energy of every iteration, checkpoints = the
model weights, launcher logs), `evaluation/` (the final weights evaluated frozen),
`results/` (tables) and `figures/`, and a `README.md` with the results.
Each run folder is named `<network>_<electrons>e_eps<ε>`, e.g. `slaternet_18e_eps5` is
SlaterNet with 18 electrons at ε = 5 (until 2026-11-02 `slaternet_eps5`; logs and
`summary.json` files written before still show the old names).
The frozen-weight evaluation itself (`analysis/evaluate_weights.py`, `analysis/observables.py`)
is shared by both systems.

The earlier run with the authors' code (6 electrons) is in `runs/archive/stage08_authors_code/`.
