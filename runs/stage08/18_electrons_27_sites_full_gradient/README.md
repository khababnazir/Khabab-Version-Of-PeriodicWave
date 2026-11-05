# runs/stage08/18_electrons_27_sites_full_gradient — Table I with the full gradient

The four runs of the paper's Table I (18 electrons on 27 sites, eps = 5 and 10, self-attention
network and SlaterNet) again, with **one** change from
`../18_electrons_27_sites_real_gradient/`: `--gradient full`.

**What differs.** The exact gradient of the energy is
2 < Re d  d log|Psi|/d theta  +  Im d  d phase/d theta >, with d = E_loc - <E_loc>. The
authors' code keeps only the first term: its local energy returns only the real part
(`external/PeriodicWave/periodicwave/pbc/hamiltonian.py`, line 234), so the second term,
which needs Im E_loc, is always zero. These runs keep it. Im E_loc is
-(1/2) div(|Psi|^2 grad phase) / |Psi|^2 in our units, the divergence of the probability
current, and the left-out term is the change of the phase's kinetic energy.

**Why.** In `../18_electrons_27_sites_real_gradient/gradient_test/` SlaterNet eps = 10,
continued from iteration 33,545 with the full gradient, did not show the energy rise of the
original run. At 150,000 its frozen-weight energy was 0.140 meV per electron lower.

**What stays the same**: seed 0 (so the same initial weights and walkers), the paper's Table II
settings (4096 walkers, 150,000 iterations, learning rate 10 / (1 + t/100000), KFAC norm
constraint 1e-3, damping 1e-3, clipping 5), the released architecture (SlaterNet with one
determinant; attention network 3 layers, 6 heads, 1 perceptron, 4 determinants), folx +
slogdet. `train_moire.py` writes all of it to each run's `settings.json`.

**This is our addition, not the reproduction**: the paper's numbers were made with the authors'
gradient, and so are ours in `../18_electrons_27_sites_real_gradient/`.

**Started 2026-11-03 05:23:54 server time (14:23 Korean time)** with
`bash learn/stage08_own_code/launch_table1.sh full`. **GPUs: both, one run each at a time**:

| GPU | queue (one after the other; each run evaluated with frozen weights right after it) |
|---|---|
| 0 | `attention_18e_eps5`, then `slaternet_18e_eps5` |
| 1 | `attention_18e_eps10`, then `slaternet_18e_eps10` |

Expected: attention about 41-43 hours, SlaterNet about 9 hours (as with the authors' gradient),
so about 2.2 days per GPU. Run names are the same as in the real-gradient folder; this
folder's name, and `"gradient": "full"` in each `settings.json`, say which gradient was used.

Folders: `training/` (runs and `logs/`), `evaluation/` (frozen weights, 5,000 steps x 4,096
walkers, burn-in 200, seed 0, as before), then `results/` and `figures/` from
`learn/stage08_own_code/analysis/18_electrons_27_sites/table1_results.py --gradient full`
and `compare_gradients.py` beside it.

## Results (2026-11-05)

All four runs finished with exit code 0 and no restarts (server time): attention eps = 10 on
2026-11-04 at 20:57 (39.6 h), attention eps = 5 at 22:34 (41.2 h), SlaterNet eps = 10 on
2026-11-05 at 07:09 (8.9 h), SlaterNet eps = 5 at 09:11 (9.3 h). Each was evaluated with
frozen weights right after it. Tables in `results/`, figures in `figures/`: 1-4 from
`table1_results.py --gradient full`, 5 from `compare_gradients.py` (both in
`learn/stage08_own_code/analysis/18_electrons_27_sites/`). Energies in meV per electron;
errors of our averages from blocking, in brackets.

**Frozen weights at 150,000** (`results/real_vs_full.csv`):

| run | authors' gradient | full gradient | full - authors' | paper Table I | full - paper |
|---|---|---|---|---|---|
| attention eps = 10 | -32.0354(2) | -32.0407(2) | -0.0053(2) | -32.070(7) | +0.029 |
| attention eps = 5 | -59.0758(2) | -59.0878(2) | -0.0120(3) | -59.127(9) | +0.039 |
| SlaterNet eps = 10 | -31.1442(6) | -31.3491(6) | -0.2049(9) | -31.35(2) | +0.001 |
| SlaterNet eps = 5 | -57.8780(7) | -58.0743(8) | -0.1963(11) | -58.01(3) | -0.064 |

Training, last 20 %: -32.0540, -59.1033, -31.3665, -58.0939 with the full gradient
(-32.0472, -59.0933, -31.1567, -57.8940 with the authors'). BP-ED (paper): -31.32443
(eps = 10), -57.80848 (eps = 5).

**Correlation energy**, E(attention) - E(SlaterNet), frozen weights
(`results/correlation_real_vs_full.csv`):

| eps | authors' gradient | full gradient | paper |
|---|---|---|---|
| 10 | -0.891(1) | -0.692(1) | -0.72 |
| 5 | -1.198(1) | -1.014(1) | -1.117 |

**Settled.** In every evaluation the two halves agree within two errors, except the first
evaluation of attention eps = 10: second half higher by 0.0008(3), 2.3 errors. (On 2026-11-05 I
first quoted 2.7 errors, from rounded numbers.) Its second evaluation, with 10x the burn-in
(2,000 steps) and seed 1 (`evaluation/attention_18e_eps10_burnin2000_seed1/`, made by
`evaluation/logs/second_eval_attention_eps10.sh`), gives the same energy, -32.0407(2), with
halves 1.2 errors apart.

**Where the electrons are** (`results/structure.csv`, figures 2-4). The attention network is
unchanged: S(kappa) 0.73 and 0.72, sublattices 1/3 each, site occupations 0.55-0.75. The
SlaterNet is a little more uneven than with the authors' gradient: S(kappa) 0.68 and 0.66
(before 0.66 and 0.60), sublattices 0.32/0.38/0.31 and 0.30/0.40/0.31 (before 0.31-0.35),
site occupations 0.23-0.94 and 0.15-0.97. Still no honeycomb in either network (a perfect one
gives S(kappa) = 4.5 and sublattices 0, 1/2, 1/2).

**What this says.**
- The full gradient lowers every energy. For the attention network the effect is small
  (0.005 and 0.012); for SlaterNet it is large (0.20 at both eps).
- SlaterNet with the full gradient does not rise during training (figure 5c). It ends level
  with the paper's SlaterNet at eps = 10 (+0.001, the paper's error is 0.02), and 0.064 below
  it at eps = 5 (2.1 of the paper's error bars).
- The attention network stays 0.029 and 0.039 above the paper (about 4 of its error bars), so
  the gradient does not explain that remaining gap.
- The correlation energy moves to -0.69 and -1.01, close to the paper's -0.72 and -1.12 (with
  the authors' gradient: -0.89 and -1.20). Most of the earlier difference came from our
  SlaterNet.

**What it does not show.**
- One seed per run. The earlier gradient test (`../18_electrons_27_sites_real_gradient/gradient_test/`)
  and these runs agree that the full gradient avoids the rise of SlaterNet eps = 10, but not on
  the value: the branch, started from the authors'-gradient state at 33,545, ended at
  -31.2838(5), 0.065 above this run from the start. Neither repeats a run with other random
  numbers.
- The paper's numbers were made with the authors' code, so with the real-part gradient. Why our
  real-part SlaterNet is 0.13-0.21 above the paper's, and theirs is not, is not known; the full
  gradient brings ours level with or below theirs, but that does not say how theirs got there.
- Only Table I (18 electrons); the 9-site runs of Fig. 5 were not repeated with the full gradient.
