# runs/stage08/table1 — 18 electrons on 27 sites (the paper's Table I), our own code

**Started 2026-10-30 12:16 server time (21:16 Korean time), all four finished 2026-11-01**
(attention 43.4 h and 41.2 h, SlaterNet 9.1 h and 8.8 h; no restarts, every value finite) with
`bash learn/stage08_own_code/launch_table1.sh`, code commit b6eb8f8.

Every run: `train_moire.py --electrons 18 --settings table2 --laplacian folx --determinants slogdet`,
that is the paper's Table II: 4096 walkers, 150,000 iterations, learning rate 10 / (1 + t/100000),
attention network 3 layers, 6 heads, 1 perceptron per layer, 4 determinants, clipping 5,
KFAC norm constraint 1e-3, damping 1e-3. SlaterNet keeps the released sizes (Table II lists none).

**GPUs: both, one run on each, 4096 walkers per run.**

| GPU | queue (one after the other) |
|---|---|
| 0 | `attention_eps5`, then `slaternet_eps5` |
| 1 | `attention_eps10`, then `slaternet_eps10` |

**Expected time (estimate).** In the learning-rate test, with both GPUs hot (85 C, clock lowered
for heat, code 0x20), one iteration took 1.05-1.13 s, against 0.82 s with GPU 0 alone. So the
attention runs take about 44-47 hours and the SlaterNet runs about 8-10 hours after them:
about 2.2-2.4 days per GPU.

**Records**, in `logs/`: `gpuN.log` (one line per 1,000 iterations: energy, acceptance, step width,
learning rate, how often the step was capped, speed), `gpuN_monitor_<date>_<time>.csv` (the GPU,
every 10 s) and `gpuN_system_<date>_<time>.csv` (CPU and memory of the machine and of the run).

## Checks before the start

All tests pass (CPU): Stage 3 (slogdet sum = det sum to 8e-14), Stage 4 (folx kinetic energy =
full Hessian to 1e-10), Stage 5, test_networks.py, test_train_moire.py (11 tests: Table II value by
value; at 27 sites our local energy equals the authors' to 6.6e-13 with folx + slogdet), and the
analysis tests. On the GPU, in a run's precision, fast and reference local energy differ by
5e-5 meV per electron on average, against a statistical error of 0.49 (speed_study/production_check.py).

**Learning-rate test** (`lr_test/`, 2,000 iterations each, attention network, eps = 5, same start, seed 0):

| | eta0 = 10 (Table II) | eta0 = 0.1 (released code) |
|---|---|---|
| all values finite, checkpoint complete | yes | yes |
| energy (meV per electron), iterations 0-99 | -40.82 | -40.86 |
| energy, iterations 1900-1999 | -58.09 | -58.29 |
| acceptance at the end | 0.53 | 0.52 |
| iterations whose step was set by KFAC's norm constraint | 100 % | 100 % |
| eta^2 g^T F^-1 g at the end (the cap is 1e-3) | 1.7e5 | 18.5 |

What it shows: in both runs KFAC's norm constraint set every step, and when it does, the
learning rate cancels out of the step exactly. So for these 2,000 iterations eta0 = 10 and
0.1 took the same steps; the runs began at the same energy (-28.2218 meV per electron) and
drifted apart by about 0.2 meV per electron only through rounding and random sampling. That
0.2 is a useful measure of run-to-run spread this early, not an effect of the learning rate.
The paper's "unusually high" eta0 = 10 keeps the step at the cap for longer.

Not checked, to start the runs at once as asked: continuing an 18-electron run from a
checkpoint (the same code is tested at 6 electrons, test_train_moire.py).

## Results (2026-11-02)

Frozen-weight evaluation of the final weights (and of SlaterNet eps = 10 at iteration 33,545):
`learn/stage08_own_code/analysis/evaluate_weights.py`, 5,000 steps x 4,096 walkers per run, one GPU
per run, both GPUs; code commit 0fdef8f. Tables and figures from `table1_results.py`, in
`results/` and `figures/`. Energies in meV per electron; errors of our averages from blocking.

| run | frozen weights | training, last 20 % | paper Table I | frozen - paper | training - paper |
|---|---|---|---|---|---|
| attention eps = 10 | -32.0354(2) | -32.0472(1) | -32.070(7) | +0.035 | +0.023 |
| SlaterNet eps = 10 | -31.1442(6) | -31.1567(3) | -31.35(2) | +0.206 | +0.193 |
| attention eps = 5 | -59.0758(2) | -59.0933(1) | -59.127(9) | +0.051 | +0.034 |
| SlaterNet eps = 5 | -57.8780(7) | -57.8940(3) | -58.01(3) | +0.132 | +0.116 |
| SlaterNet eps = 10, iteration 33,545 | -31.2667(6) | -31.2815(6) | -31.35(2) | +0.083 | +0.069 |

BP-ED (paper): -31.32443 (eps = 10), -57.80848 (eps = 5). Our attention network is 0.71 and 1.27
meV below it; our SlaterNet 0.18 above it at eps = 10 and 0.07 below at eps = 5.
Correlation energy E(attention) - E(SlaterNet), frozen weights: -0.891 (eps = 10) and -1.198
(eps = 5); paper -0.72 and -1.117. Ours is larger because our SlaterNet energies are higher.

What the numbers say:
- The frozen-weight energy is 0.012-0.018 meV higher than the training average in every run.
  The two estimates belong to different wave functions (see evaluate_weights.py).
- The paper's error bars match the spread of a single step's batch-mean energy with 4,096
  walkers (results/table1.csv, column per_step_spread: 0.0084, 0.0226, 0.0106, 0.0325 against
  the paper's 0.007, 0.02, 0.009, 0.03). Its captions of Figs. 3(b), 5 and 7 define its error
  bars that way. So the paper's numbers are most likely training averages, and our training
  averages are the like-for-like comparison: the attention network is 0.023 and 0.034 meV
  above the paper, about three of its error bars.
- SlaterNet eps = 10: the frozen energy at iteration 33,545 is 0.12 meV LOWER than at the
  end. The rise in the training curve after ~35,000 iterations is real, not a sampling effect:
  the optimiser moved the SlaterNet to a higher-energy state and stayed there.

Where the electrons are (results/structure.csv, figures 2-4):
- Attention network, both eps: every site filled almost evenly (site occupations
  0.53-0.73 against 2/3 for an even filling), the three sublattices 1/3 each, S(kappa) = 0.73.
  No honeycomb, also at eps = 5, where the paper's Fig. 6(b) shows one (electrons fixed on a
  perfect honeycomb give S(kappa) = 4.5). The same was found at 9 sites.
- SlaterNet, both eps: a striped pattern -- rows of nearly full sites (occupation up to 0.97)
  with weakly filled sites between them; sublattices 0.31-0.35, S(kappa) = 0.60-0.66. Not a
  honeycomb either.
- All patterns are the same in the first and second half of every evaluation.

Not checked: whether another seed, or more determinants for SlaterNet (ours has 1, the
released setting; the paper does not give its SlaterNet sizes), reaches the paper's
SlaterNet energies or the honeycomb.

## How to look again (from ~/moire-nnvmc)

```bash
cat runs/stage08/table1/results/table1.csv
tail -n 3 runs/stage08/table1/evaluation/logs/gpu0.log runs/stage08/table1/evaluation/logs/gpu1.log
```
