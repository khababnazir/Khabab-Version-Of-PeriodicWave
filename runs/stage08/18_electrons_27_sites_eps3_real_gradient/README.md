# runs/stage08/18_electrons_27_sites_eps3_real_gradient — the attention network at ε = 3

**Not in the paper: a diagnostic.** The paper's Fig. 6(b) shows a honeycomb crystal at eps = 5.
Our attention network shows none: in the frozen evaluations S(kappa) is 0.726 at eps = 5 and
0.732 at eps = 10 with the authors' gradient, 0.722 and 0.730 with the full gradient
(`../18_electrons_27_sites_real_gradient/README.md`, `../18_electrons_27_sites_full_gradient/README.md`).

Lowering eps only makes the Coulomb repulsion stronger: U = e^2 / (4 pi eps0 eps a_M) is
35.9 meV at eps = 5 and 59.8 meV at eps = 3 (tested in `test_train_moire.py`). For classical
point electrons kept on the moire sites, the repulsion favours the honeycomb at nu = 2/3: it
keeps the empty sites as far apart as possible. For the quantum ground state this is an
expectation, not a guarantee, and no guarantee that training finds it. So:
- **a crystal at eps = 3**: our network can form one; at eps = 5 it lands in a competing state,
  and the network form (the paper's printed tanh network, Eqs. 18-20, against the released
  one we train) or the training is the next suspect;
- **no crystal even at eps = 3**: our network or its training does not find it even where it
  should be favoured more (one run cannot show that the network cannot represent it).

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
the honeycomb's wave vector, it is 0.002. The first two are one modulation of the site
occupations seen twice: (7/9, -5/9) + (2/9, 5/9) = (1, 0), a moire reciprocal vector, which
gives the same phase on every site. The 27 mean site occupations themselves, at their 27 wave
vectors (`results/site_pattern.csv`, without such double counting), carry a pattern of 0.875
(0.093 at eps = 5, 0.105 at eps = 10): 58 % of it in the shell q = (2/3) kappa, with all three
directions comparably strong (0.10, 0.08, 0.07 per +-q pair), 25 % at |q| = |b|/3, 15 % at
|q| = 3.69 / a_M, and 0.9 % at kappa. The (2/3) kappa part alone, three directions whose wave
vectors add up to zero, would repeat every half supercell (6.75 sites and 4.5 electrons per
repeat); the other 41.5 % of the pattern does not, so the pattern as a whole has no such period.
What it is: a density modulation dominated by three directions, without the honeycomb's signal.
So at eps = 3 the network breaks the moire lattice's translation symmetry, but not into the
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
- With the stronger repulsion it made a different modulation, not the honeycomb,
  although for classical electrons on the sites a stronger repulsion favours the honeycomb more.
  Either the honeycomb is not the lowest state this network can reach for our Hamiltonian, or
  the training settles in another state and stays there. This run cannot decide it: that needs
  the energy of a honeycomb state of the same network to compare with. One run also cannot show
  that the network cannot represent a honeycomb.
- One seed, one network form (the released one, Table II sizes), the authors' gradient.

## The uneven density and the open shell (2026-11-10; corrected the same day, see Corrections)

**One electron, no interaction** (`shell_structure.py`, `results/shell_structure.csv`). A cell
of 27 sites allows 27 crystal momenta. One electron in the moire potential has these lowest-band
energies on them (band width 14.2 meV):

| shell, at k (1/a_M) | momenta | energy (meV) | filled by 18 electrons |
|---|---|---|---|
| 0 | 1 | -11.99 | 1 |
| 1.396 | 6 | -9.39 | 6 |
| 2.418 | 6 | -4.71 | 6 |
| 2.793 | 6 | -2.58 | **5 of 6: an open shell** |
| 3.694 | 6 | 0.17 | 0 |
| 4.189 (kappa) | 2 | 2.24 | 0 |

Without interaction the ground state of 18 electrons is therefore six-fold degenerate: one state
for each momentum left out, with total momentum minus that momentum. Each of them alone has an
even density, and so has a statistical mixture of them. A coherent superposition of several has
an uneven density, at the wave vectors k_i - k_j between members of the shell and only there:
|q| = 2.418, 2.793 and 3.694 -- not 1.396, and not kappa.

**Our site patterns lie at those wave vectors** (`results/site_pattern.csv`):

| eps | pattern | on those three shells | at 1.396 | at kappa |
|---|---|---|---|---|
| 10 | 0.105 | 98.8 % | 0.8 % | 0.5 % |
| 5 | 0.093 | 98.1 % | 1.9 % | 0.0 % |
| 3 | 0.875 | 99.0 % | 0.2 % | 0.9 % |

(The three shells hold 18 of the 26 wave vectors: about 69 % by chance.)

**The non-interacting picture is not the whole description.** In it no site can be filled more
than 19/27 = 0.704: all six momenta of the shell filled give 19 electrons, 19/27 on every site,
and taking one electron out lowers the density. Our largest site occupations are 0.723
(eps = 10), 0.732 (eps = 5) and 0.868 (eps = 3). And the calculation has no interaction in it.

**The trained, interacting wave functions, measured** (`momentum_sectors.py`,
`results/momentum_sectors.csv`). The full Hamiltonian, interaction included, does not change
when every electron is moved by the same lattice vector, so every wave function is a sum of
parts with a definite total momentum K, with weights w_K that add up to 1. From the newest
checkpoint of each run (its 4,096 walkers; errors from their spread, in brackets):

| run | weight on the six momenta of the 2.793 shell | each of the six | largest weights elsewhere |
|---|---|---|---|
| attention eps = 10 (released network, authors' gradient) | 0.984(10) | 0.14 - 0.19 | 0.005 (1.396, 2.418) |
| attention eps = 5 (released, authors' gradient) | 0.963(8) | 0.15 - 0.18 | 0.013 (3.694) |
| attention eps = 5 (released, full gradient) | 0.983(13) | 0.14 - 0.19 | 0.007 |
| attention eps = 5 (paper network, at 86,034) | 0.970(9) | 0.10 - 0.20 | 0.015 (2.418) |
| attention eps = 3 | 0.541(7) | 0.06 - 0.11 | 0.173 (3.694), 0.145 (2.418), 0.085 (1.396) |
| SlaterNet eps = 5 (released, authors' gradient) | 0.159(1) | 0.005 - 0.070 | 0.332 (3.694), 0.261 (2.418) |

What follows, and what does not:
- **Measured**: at eps = 10 and 5 the attention networks hold 96-98 % of their weight on the six
  total momenta that the open shell gives, spread over all six. So the shell picture does
  describe these interacting wave functions, and a coherent sum over several momenta is what
  makes their densities uneven.
- **Exact**: the Hamiltonian does not mix different K, so the energy of a wave function is the
  weighted average of the energies of its parts; and the lattice's symmetries turn the six
  momenta into each other, so the lowest energy is the same in each. Unevenness of this kind
  costs no energy by itself.
- **Exact**: a honeycomb in the density needs weight on momenta that differ by kappa (an ideal
  one that stays put has three such momenta, 1/3 each). No two momenta of the 2.793 shell differ
  by kappa, so a wave function with its weight in that shell cannot show a honeycomb in its
  density, at any eps. (A honeycomb only in the correlations is not excluded by this; S(kappa)
  of 0.72 excludes it.)
- **Not shown**: that these six momenta hold the ground state, or that the part of our wave
  function in each momentum is the best possible one. "Liquid or crystal at eps = 5" is the
  question which momenta hold the lower energy; it is not answered here.
- **Not described by this picture**: eps = 3 (46 % of the weight outside the shell) and the
  SlaterNet (spread over all shells).
- One checkpoint and 4,096 walkers per run, one seed per run.

For 6 electrons on 9 sites the shell at 2.418 is open too (5 of 6), and its differences lie at
2.418 and at kappa.

Size of the repulsion against the band width: e^2 / (4 pi eps0 eps a_M) / 14.2 meV is 1.3 at
eps = 10, 2.5 at eps = 5 and 4.2 at eps = 3.

## Corrections (2026-11-08, after an external review)

- The first version of the results called the eps = 3 pattern irregular, with "several
  unrelated wave vectors", and read that as a hint of a trapped training. Two of those wave
  vectors differ by a moire reciprocal vector and are the same modulation of the sites; on the
  sites the pattern is dominated by three directions at q = (2/3) kappa (above,
  `results/site_pattern.csv`). "Irregular" and that reading are withdrawn.
- The second version (same day) called it a regular triangular modulation with half the
  supercell's period. Only its (2/3) kappa part, 58.5 %, has that period; 41.5 % lies in other
  shells. Reworded after a fourth review.
- The plan said that no crystal at eps = 3 would mean that the network or its training "cannot
  form it, whatever the physics". One run cannot show that; reworded.
- A lower eps favours the honeycomb for classical point electrons on the sites; for the quantum
  state, and for what training finds, it was an expectation, not a guarantee. Reworded.
- Checked and unchanged: the recorded settings differ from `attention_18e_eps5` of the
  real-gradient folder only in eps; S(kappa) is 0.6314 in the frozen evaluation and 0.633 with
  the last checkpoint's walkers.
- 2026-11-10, after a fifth review of the open-shell section's first version (commit 5a58bc8):
  - It said "any mixture of them is an equally good ground state, and a mixture has an uneven
    density". A statistical mixture of the momentum states has an even density; only a coherent
    superposition is uneven. And the calculation had no interaction in it, so "equally good
    ground state" of the interacting system was not established. Rewritten; the interacting
    wave functions are now measured (`momentum_sectors.py`).
  - It did not say that the non-interacting picture limits a site's occupation to 19/27 = 0.704,
    which our networks exceed at all three eps (0.723, 0.732, 0.868). Added.
  - It called the unevenness "not an error of the wave function" and the states "the liquid
    branch". What is shown: the weight is on the open shell's momenta (96-98 % at eps = 10 and
    5), and unevenness of that kind costs no energy by itself. Not shown: that these are ground
    states.
