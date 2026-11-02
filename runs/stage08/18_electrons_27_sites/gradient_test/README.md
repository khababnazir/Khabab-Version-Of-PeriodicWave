# 18 electrons, 27 sites — gradient test: SlaterNet ε = 10 continued from iteration 33,545

**Question.** The training energy of SlaterNet ε = 10 (`../training/slaternet_18e_eps10`) rose
after about 35,000 iterations, and its frozen-weight energy at iteration 33,545 is 0.12 meV per
electron lower than at the end (`../README.md`). The authors' gradient (`--gradient real`) leaves
out the term 2 < Im d  d phase/d theta > of the exact energy gradient. At the end of that run
the left-out term was 0.594 of the kept one; at iteration 33,545 it was 0.088. Does the
left-out term cause the rise?

**Design** (the conditions agreed on 2026-11-02):
- Two new branches of `../training/slaternet_18e_eps10`, which is only read. Both start from its
  `checkpoint_033545.npz`, the only checkpoint in each branch folder, so neither can continue
  from another one (`learn/stage08_own_code/branch_run.py make`).
- The whole saved state comes back: weights, KFAC state (its step counter included), walkers,
  random key, step width and iteration counter. Checked before the runs, below.
- Only the gradient differs. Each `settings.json` is the original's with `"gradient"` changed,
  and `train_moire.py` refuses to continue if anything else differs: architecture, batch,
  sampling and norm constraint are the original's.
- **GPU 1: `real`** (as the original, which also ran on GPU 1). **GPU 0: `full`.** One run per
  GPU, 4096 walkers each.
- Both to 76,070 iterations, the original's first checkpoint after the rise, so that the
  real branch can be compared with the original checkpoint to checkpoint.
- Then each run is compared with the original row by row, and evaluated with frozen weights
  exactly as the Table I runs were (`evaluate_weights.py`, 5,000 steps x 4,096 walkers,
  burn-in 200, seed 0). If an evaluation is still drifting, a second one follows with a longer
  burn-in and another seed.
- Reading: if real rises again while full stays lower, that supports the left-out term as a
  cause. If both rise, the next test is the KFAC step limit (norm constraint). Neither outcome
  is assumed.

Started with `bash learn/stage08_own_code/launch_gradient_test.sh` on 2026-11-02 09:19 server
time (18:19 Korean time), code commit fff7d51. Expected: about 2.5 hours of training, then about
20 minutes of evaluation, on each GPU.

## Restoration check (`restore_check/`, 2026-11-02 09:16)

`bash learn/stage08_own_code/launch_gradient_test.sh check`: 20 iterations of each branch
(33,545-33,564), then `branch_run.py compare` against the original's rows:

| branch | rows identical to the original | difference in row 33,545 |
|---|---|---|
| real (GPU 1) | 20 of 20, every column, bit for bit | none |
| full (GPU 0) | 0 of 20 | none in energy, variance, Im E, acceptance, step width, learning rate; `fisher_step_sq` 294 (of about 9,900) |

So a restart restores the whole state, and the GPU computes the same numbers again. Row 33,545
is measured before the first update, so with the full gradient only `fisher_step_sq` differs
there (KFAC's step size, which depends on the gradient). After that update the two runs
separate: the largest energy difference in 20 iterations is 0.254 code units, 0.048 meV per
electron.

## Folders

| Folder | What it holds |
|---|---|
| `restore_check/` | the two 20-iteration branches of the check, and their logs |
| `training/` | the two runs: `settings.json`, `branch.json` (where the branch came from), `train_stats.csv`, checkpoints, `starts.log`; and `logs/` (`gpuN.log`: progress, comparison with the original, evaluation; GPU and system records every 10 s) |
| `evaluation/` | the frozen-weight evaluations, after the runs |

In each `train_stats.csv` the rows before 33,545 are copied from the original, unchanged; the
branch's own rows follow. Their `seconds` column counts from the branch's start.

## Results (2026-11-02)

Both runs finished with exit code 0 and no restarts: real (GPU 1) at 11:52, full (GPU 0) at
11:59 server time. Their evaluations finished at 12:10 and 12:18, the second evaluations at
12:45. Tables in `results/`, figure `figures/gradient_test.png`, both made by
`learn/stage08_own_code/analysis/18_electrons_27_sites/gradient_test_results.py`. Energies in
meV per electron; errors of the last digits, from blocking, in brackets.

**1. The real branch is the original run, exactly.** All 42,525 rows of its `train_stats.csv`
from 33,545 to 76,069 equal the original's in every column, and its `checkpoint_076070.npz` is
byte for byte the original's (`results/comparison.json`). So it repeats the original's rise
exactly. It is a replay of the same computation, not a repeat with other random numbers.

**2. Training**: the mean of the 2,000 iterations before each of the original's checkpoints
(`results/training_at_original_checkpoints.csv`; every 1,000 iterations in
`results/training_windows.csv`).

| original's checkpoint | original = real branch | full branch | full - original |
|---|---|---|---|
| 42,076 | -31.2768(11) | -31.2896(9) | -0.013 |
| 50,580 | -31.2535(8) | -31.2918(10) | -0.038 |
| 59,089 | -31.2044(10) | -31.2928(9) | -0.089 |
| 67,579 | -31.1706(13) | -31.2940(9) | -0.123 |
| 76,070 | -31.1693(10) | -31.2949(7) | -0.126 |

From the first 2,000 iterations after the branch point to the last 2,000 before 76,070, the
original (and so the real branch) rises by 0.1143, from -31.2836(10) to -31.1693(10). The
full branch falls by 0.0086, from -31.2863(11) to -31.2949(7).

**3. Frozen weights**, 5,000 steps x 4,096 walkers each (`results/frozen_energies.csv`):

| weights | burn-in 200, seed 0 | burn-in 2,000, seed 1 |
|---|---|---|
| original, iteration 33,545 (the branch point) | -31.2667(6) | not run |
| real branch, 76,070 | -31.1529(7) | -31.1528(5) |
| full branch, 76,070 | -31.2782(6) | -31.2785(6) |
| original, 150,000 (its end) | -31.1442(6) | not run |

- Full minus real at 76,070: -0.1253(9) in the first evaluations, -0.1257(8) in the second.
- The full branch at 76,070 is also 0.0115(8) below the original at the branch point.
- Settled: in every evaluation the two halves agree within two errors. The closest was the
  full branch's first evaluation (second half higher by 0.0023(12), 1.9 errors), so the second
  evaluation, with 10x the burn-in and seed 1, was run for both branches. It agrees with the
  first within 0.0003 for both.
- As in the Table I runs, each frozen energy lies above the training mean just before its
  checkpoint: by 0.016 (real) and 0.017 (full).

**What this says.** The outcome is the first of the two named beforehand. The real branch rose
again (exactly as the original, being the same computation), while the full branch stayed low
and kept falling slowly. That supports the left-out term 2 < Im d  d phase/d theta >
contributing to the rise of SlaterNet eps = 10.

What it does not show:
- From the first update on, the two branches differ not only in the gradient but also in their
  random path: walkers and parameters move differently. With one branch per gradient, "the
  full gradient prevents the rise" cannot be told apart from "this one path happened not to
  rise". A real-gradient branch from 33,545 with a different random key would separate the
  two; it was not run, as it is not part of the agreed test.
- The test ran only to 76,070. Whether the full branch stays low to 150,000, and where it ends,
  is not known.
- It covers only SlaterNet eps = 10. The attention network and eps = 5 were not tested.
- The reproduction keeps the authors' gradient (decision of 2026-11-02), so the Table I values
  in `../results/` are unchanged. For comparison only: the full branch at 76,070 is 0.072
  above the paper's SlaterNet -31.35(2), where our final value is 0.206 above it.
