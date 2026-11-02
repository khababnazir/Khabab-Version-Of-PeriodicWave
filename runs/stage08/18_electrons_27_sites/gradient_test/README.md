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

## Results

After the runs.
