# runs/stage08/table1 — 18 electrons on 27 sites (the paper's Table I), our own code

**Started 2026-10-30 12:16 server time (21:16 Korean time)** with
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

## Tomorrow: how to look (from ~/moire-nnvmc)

```bash
tail -n 3 runs/stage08/table1/logs/gpu0.log runs/stage08/table1/logs/gpu1.log
wc -l runs/stage08/table1/attention_eps*/train_stats.csv      # iterations done + 1
nvidia-smi
```

Paper's Table I, meV per electron: self-attention network -59.127(9) at eps = 5, -32.070(7) at
eps = 10; SlaterNet -58.01(3) and -31.35(2).

If a run has stopped (for example a GPU dropped off the bus): start the launcher again. Each run
continues from its newest checkpoint (written every 30 minutes), a finished run exits at once,
and a queue whose GPU is still busy exits with code 75 without starting a second copy.
