# runs/stage08/18_electrons_27_sites_eps3_real_gradient — the attention network at ε = 3

**Not in the paper: a diagnostic.** The paper's Fig. 6(b) shows a honeycomb crystal at eps = 5.
Our attention network shows none: in the frozen evaluations S(kappa) is 0.726 at eps = 5 and
0.732 at eps = 10 with the authors' gradient, 0.722 and 0.730 with the full gradient
(`../18_electrons_27_sites_real_gradient/README.md`, `../18_electrons_27_sites_full_gradient/README.md`).

Lowering eps only makes the Coulomb repulsion stronger: U = e^2 / (4 pi eps0 eps a_M) is
35.9 meV at eps = 5 and 59.8 meV at eps = 3 (tested in `test_train_moire.py`). For electrons
kept on the moire sites, the repulsion favours the honeycomb at nu = 2/3: it keeps the empty
sites as far apart as possible. So:
- **a crystal at eps = 3**: our network can form one; at eps = 5 it lands in a competing state,
  and the network form (the paper's printed tanh network, Eqs. 18-20, against the released
  one we train) or the training is the next suspect;
- **no crystal even at eps = 3**: our network or its training cannot form it, whatever the physics.

Too low an eps could also pull the electrons off their sites, where other arrangements compete;
eps = 3 is a moderate step. There is no paper value and no BP-ED at eps = 3.

**One run, `attention_18e_eps3`, the same as `attention_18e_eps5` in
`../18_electrons_27_sites_real_gradient/` except eps**: seed 0 (the same initial weights and
walkers), the authors' gradient (`--gradient real`), Table II (4096 walkers, 150,000
iterations, learning rate 10 / (1 + t/100000), KFAC norm constraint 1e-3, damping 1e-3,
clipping 5), the released network family with Table II sizes (3 layers, 6 heads, 1
perceptron, 4 determinants), folx + slogdet. `settings.json` records it.

**Started 2026-11-06 06:01:36 server time (15:01 Korean time)** with
`bash learn/stage08_own_code/launch_table1.sh eps3` (code of commit 7b557b3). **GPU 0 only**;
GPU 1 stays free. The first 481 iterations took 0.85 s each, so about 35 hours, then the
frozen-weight evaluation (about 1.3 hours; 5,000 steps x 4,096 walkers, burn-in 200, seed 0).

**Watching for the crystal during training** (CPU, safe while the run trains):

```bash
cd learn/stage08_own_code/analysis/18_electrons_27_sites
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= uv run --no-project --python ../../../../.venv/bin/python \
    --with matplotlib --with pyblock python checkpoint_structure.py \
    ../../../../runs/stage08/18_electrons_27_sites_eps3_real_gradient/training/attention_18e_eps3 \
    ../../../../runs/stage08/18_electrons_27_sites_real_gradient/training/attention_18e_eps5 \
    ../../../../runs/stage08/18_electrons_27_sites_real_gradient/training/attention_18e_eps10
```

It writes `results/checkpoint_structure_<run>.csv` beside each run and
`figures/checkpoint_structure.png` here. A honeycomb shows as S(kappa) rising well above 1
(4.5 for electrons fixed on a perfect one). If the sampling also stays in one of the three
honeycombs, the emptiest sublattice's share falls from 1/3 towards 0.

Folders: `training/` (the run and `logs/`), `evaluation/` (frozen weights), then `results/`
and `figures/`.

## Results

After the run.
