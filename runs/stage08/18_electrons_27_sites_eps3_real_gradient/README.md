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

## Results (2026-11-08)

The run ended on 2026-11-07 at 18:14:37 server time (exit code 0, 36.2 hours, 73 checkpoints),
its frozen-weight evaluation at 19:22:35 (exit code 0, 1.1 hours; sanity check passed: first
evaluation steps -511.307, last training steps -511.429 code units). Tables in `results/`,
figures in `figures/`, made by `eps3_results.py` (tables, figures 1 and 2) and
`checkpoint_structure.py` (`checkpoint_structure*`), both in
`learn/stage08_own_code/analysis/18_electrons_27_sites/`. The eps = 5 and 10 runs compared
with are the authors'-gradient ones in `../18_electrons_27_sites_real_gradient/`.

**Energy** (meV per electron, blocking errors in brackets; `results/energies.csv`): frozen
weights -95.8838(2), training last 20 % -95.9056. Kinetic 22.81, moire -25.80, Coulomb -92.89.
Settled: the two halves of the evaluation agree (-0.05 errors apart). The paper has no value at
eps = 3.

**No honeycomb crystal** (`results/structure.csv`):

| eps | S(kappa) | connected part | share on the three sublattices | site occupation |
|---|---|---|---|---|
| 10 | 0.732 | 0.732 | 0.331 / 0.334 / 0.335 | 0.54-0.72 |
| 5 | 0.726 | 0.726 | 0.334 / 0.333 / 0.333 | 0.53-0.73 |
| 3 | 0.631 | 0.629 | 0.329 / 0.343 / 0.328 | 0.43-0.87 |

S(kappa) is lower at eps = 3, not higher (a honeycomb of electrons fixed on the sites gives
4.5). At the checkpoints it rose slowly from 0.58 to 0.63 and never came near 1
(`results/checkpoint_structure_attention_18e_eps3.csv`, `figures/checkpoint_structure.png`).
During the evaluation the three sublattices stay at about 1/3 each (figure 2).

**But the density is no longer even.** Some sites are filled 87 % of the time, others 43 %
(figure 1c). The static density modulation |<rho_q>|^2 / N, summed over all wave vectors but
the moire lattice's own, is 3.41, against 0.31 at eps = 10 and 0.34 at eps = 5
(`results/density_modulation.csv`). Its largest parts, 0.14 each, are at
m = +-(7/9, -5/9) and +-(2/9, 5/9), then 0.12 at +-(2/3, -2/3) (q = m1 b1 + m2 b2); at kappa,
the honeycomb's wave vector, it is 0.002. So at eps = 3 the network breaks the moire lattice's
translation symmetry, but into a pattern with several unrelated wave vectors, not into the
honeycomb.

**The electrons stay on their sites** (`results/off_site.csv`). At all three eps 0.26 of the
density lies more than 0.35 a_M from the nearest site, and about 0.06 within 0.15 a_M of the
centres of the 'down' triangles, the potential's shallower minima (-23.3 meV for one electron,
against -63.6 meV on a site). The faint Y shapes in figure 1c are this density; they are not new
at eps = 3. Before the run we worried that a stronger repulsion might pull the electrons off
their sites; at eps = 3 it does not.

**What this says.**
- Our network can make an uneven density: at eps = 3 it breaks the translation symmetry
  strongly. So the missing honeycomb at eps = 5 is not because it can only make even ones.
- With the stronger repulsion it made an irregular pattern, not the honeycomb, although for
  electrons kept on the sites a stronger repulsion should favour the honeycomb more. Either the
  honeycomb is not the lowest state this network can reach for our Hamiltonian, or the training
  settles in another state and stays there. A pattern with several unrelated wave vectors looks
  more like the second, but this run cannot decide it: that needs the energy of a honeycomb
  state of the same network to compare with.
- One seed, one network form (the released one, Table II sizes), the authors' gradient.
