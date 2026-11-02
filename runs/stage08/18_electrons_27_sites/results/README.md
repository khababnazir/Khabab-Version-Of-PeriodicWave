# 18 electrons, 27 sites — results: our numbers for the paper's Table I

Written by `learn/stage08_own_code/analysis/18_electrons_27_sites/table1_results.py`.
Energies in **meV per electron**; errors of our averages are one standard error from blocking.
Every table has the columns `electrons` (18) and `sites` (27). `run` names an evaluation
folder in `../evaluation/`; `slaternet_eps10_at_033545` is SlaterNet ε = 10 after 33,545
iterations.

## table1.csv — energies, one row per evaluation

| Column | Meaning |
|---|---|
| `checkpoint` | iterations done by the weights evaluated |
| `E_frozen`, `_error` | energy of those weights, frozen (5,000 steps × 4096 walkers) |
| `E_training_last20pct`, `_error` | mean energy over the last 20 % of training before that checkpoint |
| `E_paper_table1`, `_error` | the paper's Table I value and its printed error |
| `frozen_minus_paper`, `_error`, `_in_errors` | our frozen energy minus the paper's, its error, and the ratio |
| `training_minus_paper` | our training mean minus the paper's (the like-for-like comparison, see `../README.md`) |
| `E_bp_ed`, `frozen_minus_bp_ed` | the paper's BP-ED energy, and ours minus it |
| `per_step_spread`, `training_per_step_spread` | spread of ONE step's batch-mean energy (frozen, training); the paper's error bars match it |
| `kinetic`, `moire`, `coulomb` | the frozen energy split into its three parts |
| `sigma_local_energy_meV` | spread of the local energy over the walkers, meV, whole system |
| `pmove`, `sanity_first_200_steps_meV`, `configurations`, `steps_per_second` | acceptance, the sanity check's value, how much was sampled, speed |

## structure.csv — where the electrons are

| Column | Meaning |
|---|---|
| `S_kappa`, `S_kappa_connected` | structure factor at the corner κ of the moiré Brillouin zone (above 1 for a honeycomb; 4.5 for electrons fixed on a perfect one), and without its static part |
| `S_kappa_first_half`, `_second_half` | the same over each half of the evaluation |
| `sublattice_0`, `_1`, `_2` | fraction of the electrons on each of the three sublattices (sites with n1 + n2 = 0, 1, 2 mod 3): 1/3 each for a liquid, (0, 1/2, 1/2) for a honeycomb that stays put |
| `site_occupation_min`, `_max` | least and most filled site (2/3 each if even) |

## correlation_energy.csv

E(self-attention) − E(SlaterNet) at each ε, final weights frozen: `ours`, `ours_error`, and the
paper's (from its Table I).
