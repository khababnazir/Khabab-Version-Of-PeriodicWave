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
`learn/stage08_own_code/analysis/18_electrons_27_sites/table1_results.py --gradient full`.

## Results

After the runs.
