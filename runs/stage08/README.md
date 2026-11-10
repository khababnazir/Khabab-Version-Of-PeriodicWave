# runs/stage08 — data of our own Stage 8

Two systems, trained with our code (`learn/stage08_own_code/`), one folder per system and
gradient convention:

- `_real_gradient`: the authors' convention, as in their code. Their local energy keeps only
  its real part (`external/PeriodicWave/periodicwave/pbc/hamiltonian.py`, line 234), so the
  gradient term 2 < Im d  d phase/d theta > is left out. This is the reproduction of the paper.
- `_full_gradient`: the same runs with that term kept (`train_moire.py --gradient full`),
  everything else identical. Our addition, run 2026-11-03 to 11-05; results in its README.
- `18_electrons_27_sites_eps3_real_gradient/`: not in the paper. The attention network at
  eps = 3 (a stronger repulsion) with the authors' gradient: does it form the honeycomb crystal
  that ours does not form at eps = 5? Run 2026-11-06 to 11-07: no honeycomb, but an uneven
  density (dominated by three directions, not the honeycomb); results in its README.
- `18_electrons_27_sites_paper_network_real_gradient/`: the attention network as the paper
  prints it (`--network-form paper`: tanh, no layer norm) at eps = 5 with the authors' gradient,
  against our released-network run: does it form the honeycomb? Started 2026-11-08.
- `18_electrons_27_sites_authors_code/`: a reference. The authors' own program, unchanged and
  started with their own script (its settings, not Table II; no line of ours in the training),
  for 18 electrons at eps = 5 and 10. Started 2026-11-10.

| Folder | System | Gradient | Paper | Analysis scripts |
|---|---|---|---|---|
| `6_electrons_9_sites_real_gradient/` | 6 electrons on 9 sites | authors' (real) | Fig. 5 | `learn/stage08_own_code/analysis/6_electrons_9_sites/` |
| `18_electrons_27_sites_real_gradient/` | 18 electrons on 27 sites | authors' (real) | Table I, Fig. 6 | `learn/stage08_own_code/analysis/18_electrons_27_sites/` |
| `18_electrons_27_sites_full_gradient/` | 18 electrons on 27 sites | full | Table I, Fig. 6 | the same, with `--gradient full` |
| `18_electrons_27_sites_eps3_real_gradient/` | 18 electrons on 27 sites, eps = 3 | authors' (real) | none (our diagnostic) | the same; `checkpoint_structure.py` |
| `18_electrons_27_sites_paper_network_real_gradient/` | 18 electrons on 27 sites, eps = 5, the paper's printed network | authors' (real) | Fig. 6 | the same; `checkpoint_structure.py` |
| `18_electrons_27_sites_authors_code/` | 18 electrons on 27 sites, eps = 5 and 10, the AUTHORS' program and script, unchanged | theirs | Table I, Fig. 6 | to come |

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
