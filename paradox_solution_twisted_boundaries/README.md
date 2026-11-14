# Paradox experiment: twisted boundary conditions

Started 2026-11-14.  Everything about this experiment is in this folder.  Nothing in `learn/` was
changed: the two files that had to change were **copied** here and changed here, and
`CHANGES.diff` shows every changed line.

## The paradox

The paper's Fig. 6 shows, at eps = 5, a honeycomb crystal (18 electrons on 27 sites: two of
every three sites filled).  No run of ours shows it, and neither do the reference runs with the
authors' own code (`runs/stage08/18_electrons_27_sites_authors_code*/`).  The energies agree
with the paper to a few hundredths of a meV; the pattern does not.

Our diagnosis so far (`shell_structure.py`, the open-shell report): in the 27-site supercell with
**periodic** boundary conditions, 18 electrons without interaction do not have a unique lowest
state.  The last 5 electrons go into a shell of 6 levels of equal energy -- an **open shell**.
At eps = 5 and 10 the trained networks keep 96-98 % of their weight on the six total momenta of
that open shell.

## The idea tested here

Change the boundary condition so that the shell is **closed**, and see what the same network,
trained the same way, does then.  A twisted boundary condition means: when one electron goes
once around the supercell along L1 or L2, the wave function comes back multiplied by a phase,

    Psi(..., r_j + L_i, ...) = exp(2 pi i T_i) Psi(..., r_j, ...)        (periodic: T_1 = T_2 = 0)

This shifts every allowed one-electron momentum from k to k + k_twist, k_twist = T_1 G_1 + T_2 G_2.
With T = (1/3, 1/3) the 18 lowest levels are 3 + 3 + 6 + 6, separated from the 19th by a gap of
1.126 meV (`code/twist_shells.py`; periodic: gap 0, shells 1 + 6 + 6 + 5 of 6).  The only twists
that keep the momentum grid three-fold symmetric are (0, 0), (1/3, 1/3) and (2/3, 2/3); the last
two are mirror images of each other (k -> -k) and have the same levels.

In the code the twist is one line: the network's periodic wave function is multiplied by
exp(i k_twist . (r_1 + ... + r_N)).  Only the **phase** changes.  The sampling uses |Psi|^2 and
does not see it; the kinetic energy differentiates the phase and does.

Two things to keep in mind:
- The shifted momentum grid is no longer symmetric under k -> -k.  A twisted wave function
  cannot be real, so this run uses `--gradient full` (the imaginary part of the local energy
  kept), and its baseline is the periodic **full-gradient** run.
- "Closed shell" is a statement about electrons **without** interaction.  The gap is 1.1 meV; the
  repulsion between neighbouring sites is about 36 meV at eps = 5.  Whether the gap matters for
  the interacting state is what the run tests.

## What is in this folder

    README.md              this file
    CHANGES.diff           every changed line of the two copied files, against learn/ at commit b6f773f
    code/
      train_moire.py       COPY of learn/stage08_own_code/train_moire.py + the option --twist T1 T2
      evaluate_weights.py  COPY of learn/stage08_own_code/analysis/evaluate_weights.py; only its
                           import path changed, so that it uses the train_moire.py above
      twist_shells.py      new: the one-electron levels with and without a twist (numpy only)
      test_twist.py        new: the checks (below)
      twist_mutants.py     new: plants mistakes to see whether the checks notice them
      launch_twisted.sh    new: starts the run
    results/one_electron_levels.csv     written by twist_shells.py
    runs/18_electrons_27_sites_twist_one_third_full_gradient/
      training/attention_18e_eps5/      settings.json, train_stats.csv, checkpoints, starts.log
      training/logs/                    gpu0.log, GPU and CPU/memory monitors (every 10 s), watch.log
      evaluation/attention_18e_eps5/    the final weights, frozen: energies.csv, observables.npz, summary.json

Used **unchanged, from `learn/`** (not copied): Stages 3-7 (wave function, sampling, local
energy, gradient, Ewald sum), `observables.py`, `system_monitor.py`, `watch_run.py`.

What `CHANGES.diff` contains, in words.  `train_moire.py`: two settings (`twist_1`, `twist_2`,
default 0), two small functions (`twist_vector`, `add_twist`), three lines in `make_problem`
that apply the twist when it is not zero, the command-line option, and one line so that folders
made by the original code count as periodic.  `evaluate_weights.py`: the two import-path lines.
The rest is explanation text.

## The run

    cd ~/moire-nnvmc/paradox_solution_twisted_boundaries/code
    bash launch_twisted.sh            # GPU 0 only; about 44 hours of training + 1.3 hours of evaluation
    bash launch_twisted.sh watch      # the watchdog, once train_stats.csv exists

which runs

    train_moire.py --electrons 18 --settings table2 --laplacian folx --determinants slogdet \
        --network attention --eps 5 --gradient full --twist 1/3 1/3

That is the paper's Table II (4096 walkers, 150,000 iterations, learning rate 10 / (1 + t/100000),
3 layers, 6 heads, 4 determinants), the released network form, seed 0.  **Everything except the
twist is identical to the baseline**
`runs/stage08/18_electrons_27_sites_full_gradient/training/attention_18e_eps5`
(`test_twist.py` compares the stored settings entry by entry).

Baseline results (periodic, full gradient, frozen weights): energy -59.0878(2) meV per electron;
S(kappa) = 0.722 (a perfect honeycomb gives 4.5); weight on the six open-shell total momenta
0.983(13).

## Checks before the run

`cd code && ../../.venv/bin/python test_twist.py` -- 18 tests, CPU, about 3 minutes, all pass:

- k_twist . L_i = 2 pi T_i;
- moving **any** one electron by L_1 or L_2 leaves |Psi| unchanged and multiplies Psi by
  exp(2 pi i T_i): attention network at 6 and at 18 electrons, SlaterNet, several twists;
- without a twist the copy **is** the original: log Psi and the local energy are identical
  number for number, and a small training run writes the same checkpoint and train_stats.csv;
- an exact answer: a determinant of plane waves with the twist has kinetic energy
  1/2 sum |k_j + k_twist|^2 at every configuration, through both of Stage 4's routes;
- on the network: twisted minus periodic local energy equals
  1/2 N |k|^2 + k . sum grad(phase) - i k . sum grad(log|Psi|), and log|Psi| is unchanged;
- the launcher's exact command line stores the twist; another twist in the same folder is
  refused; the evaluation copy imports this folder's trainer and gives the run its twist;
- `twist_shells.py` agrees with learn's `shell_structure.py`.

Do the tests notice mistakes?  `../../.venv/bin/python twist_mutants.py` plants 12 mistakes, one
at a time, in a copy of `code/` and runs the tests on each (about 15 minutes).  All 12 were
noticed: the sign of the phase; the twist on one electron only; G_1 and G_2 swapped; the twist
never applied; applied only if both twists are non-zero; the command line dropping the twist;
the command line swapping T1 and T2; old folders not counted as periodic; the evaluation
importing learn's trainer (which knows no twist); the launcher without `--twist`; the launcher
with `--gradient real`; a wrong constant in `twist_shells.py`.

Smoke test on GPU 0 (30 iterations with the launcher's options, then a short evaluation, in
/tmp): both ended with exit code 0; 0.76 s per iteration and 18.4 GB, as the baseline at its start.
At step 0 the acceptance equals the baseline's to all digits (0.1625976711511612: the same
walkers, as it must be), and the energy is higher by 5.37 code units (baseline -150.491,
twisted -145.121).  Expected: 1/2 N |k_twist|^2 = 5.85 plus the walkers' mean of
k . sum grad(phase), whose value is not known beforehand; the test above checks that formula
configuration by configuration.

## Predictions, written down before the run (2026-11-14)

What will be measured on the final weights, with the same functions as for the baseline: the
energy, S(kappa), the 27 site occupations, the three sublattice shares, and the weight of the
state on each total momentum.

1. **If the network follows the closed shell:** nearly all weight on ONE total momentum
   (K = 0, the momentum of the 18 filled levels; the baseline has its weight spread over six),
   all 27 site occupations equal to 2/3 within the noise, S(kappa) below 1 as in the baseline.
   No honeycomb, and no uneven site pattern either.
2. **If a honeycomb crystal forms (the paper's Fig. 6):** S(kappa) well above 1 (4.5 if perfect);
   sublattice shares near 0, 1/2, 1/2 if the pattern stays in place, or even shares with a
   large S(kappa) if it moves; weight on three total momenta K0, K0 + kappa, K0 - kappa.
3. **If nothing changes:** uneven site occupations as in the periodic runs and weight on
   several total momenta.  Then the open shell of the non-interacting electrons was not the reason.

What the result can and cannot say:
- The twisted energy is NOT comparable with the paper's Table I, which is periodic.  Without
  interaction the twist alone raises the energy by 0.047 meV per electron (-6.0338 against
  -6.0809).  How much the interacting energy changes with the twist is itself a result: a
  crystal that is pinned to the lattice should barely notice the boundary condition, a liquid
  notices it.
- Outcome 1 would support the open-shell diagnosis of OUR runs.  It would not explain why the
  paper shows a honeycomb.
- One run, one seed.

## Results

(to be written when the run has finished)
