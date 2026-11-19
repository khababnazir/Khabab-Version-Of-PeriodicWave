# authors_exact_code_runs/runs/18_electrons_27_sites_eps3_seed42 — eps = 3: a honeycomb test and a second random start

Asked for by Khabab on 2026-11-19. Two runs of the authors' code (`external/PeriodicWave`, commit
`6525d2f3eec53f085b55d485e1bff45d61ef1782`, no file of it changed), their released settings,
18 electrons on 27 sites, eps = 3, the self-attention network (`CustomPsiformer`), both from the
**same initial weights and walkers** (fixed seed 42):

| | GPU 0: honeycomb test (`honeycomb_field/`) | GPU 1: ordinary training (`ordinary/`) |
|---|---|---|
| iterations 0 to 19,999 | their Hamiltonian + a field that empties one sublattice | their Hamiltonian |
| at 19,999 | checkpoint saved; S(kappa) and sublattice shares measured: did a honeycomb form? | |
| iterations 20,000 to 199,999 | their Hamiltonian, the field off, continuing from that checkpoint | their Hamiltonian |
| afterwards | final weights frozen, field off (evaluate_authors_weights.py) | the same |

The questions: does a prepared honeycomb survive once the field is gone, and is its energy (field
off, frozen weights) lower than the liquid-like state? Does ordinary training from another random
start find a honeycomb? Reference: the first eps = 3 run (`../18_electrons_27_sites_eps3_released_settings/`,
clock seed), frozen weights -95.9339(2) meV per electron, S(kappa) 0.666.

What the outcomes can and cannot say: a honeycomb that survives and is reliably LOWER in energy shows
that this network and this Hamiltonian hold a better ordered state that ordinary training missed.
One that survives but is higher is a local minimum. One that melts shows nothing about energies
(the field may have left a poor network, or the optimiser may leave a shallow minimum). One seed.

## What is changed (changes.diff)

- `run_2deg_triangular_potential_seed42.py`: their script with ONE line added,
  `cfg.debug.deterministic = True`. Their train.py then uses the seed 42 instead of a seed taken from
  the clock (train.py lines 368-372). That seed makes the initial weights (`network.init`), the
  initial walkers and every Metropolis move. Used by GPU 1 and by GPU 0 from iteration 20,000.
- `run_2deg_triangular_potential_seed42_field.py`: that copy with three more lines: the Hamiltonian
  named by `cfg.system.make_local_energy_fn` is `honeycomb_field.local_energy`, its strength is 0.5,
  and `cfg.optim.iterations = 20000` (their code saves a checkpoint at the last iteration).
- `honeycomb_field.py` (new): THEIR local energy, built unchanged, plus
  E_field = lambda * sum over electrons of sum_j cos(kappa_j . r), kappa_j the three corners of the
  moire Brillouin zone at 0, 120, 240 degrees. On the 27 sites this is 3 lambda on sublattice 0 and
  -1.5 lambda on the others: sublattice 0 is raised by 4.5 lambda = 7.6 meV (lambda = 0.5 code
  units = 1.69 meV); a full honeycomb gains 2.5 meV per electron from it. The walkers are sampled
  from |Psi|^2 alone; the field acts through the local energy that training lowers.

`test_honeycomb_field.py` checks, before the start: the field's values on the sites and its
periodicity; that the field script's local energy, built as their train.py builds it, is the plain
one plus exactly the field term; that the seed-42 configuration differs from the first eps = 3 run's
`config.json` only in `debug.deterministic`, and the field configuration from the seed-42 one only
in the three field entries; that both start from the same weights.

## How it was started

```bash
cd ~/moire-nnvmc/authors_exact_code_runs/runs/18_electrons_27_sites_eps3_seed42
setsid nohup flock -n ~/moire-nnvmc/runs/gpu_locks/gpu0.lock bash how_this_was_started.sh 0 field >> logs/gpu0.log 2>&1 < /dev/null &
setsid nohup flock -n ~/moire-nnvmc/runs/gpu_locks/gpu1.lock bash how_this_was_started.sh 1 ordinary >> logs/gpu1.log 2>&1 < /dev/null &
```

Output, written by their code, in `ordinary/`, `honeycomb_field/phase1_field_on/` and
`honeycomb_field/phase2_field_off/`, each `results/2deg-CoulombMoire/CustomPsiformer/el18_0_N27_V4.44370908_45.0_U17.70578894/`
(`config.json`, `train_stats.csv`, `qmcjax_ckpt_<step>.npz` every 30 minutes). The phase-1 check is in
`honeycomb_field/phase1_check.csv` and `logs/gpu0.log`. Note: while the field is on, the energies in
phase 1's `train_stats.csv` include the field term.

## Log

- 2026-11-19, before the start: test_honeycomb_field.py, 6 tests, all pass (CPU). Three mutants, all
  caught: the field term removed from local_energy (EnergyTest fails), kappa of half the length
  (FieldTest fails), the seed line removed from the seed script (ConfigTest fails).
  check_honeycomb.py on the final checkpoint of the first eps = 3 self-attention run (1024 walkers):
  S(kappa) 0.712, sublattice shares 0.328 0.332 0.340 (its frozen evaluation: 0.666 and
  0.325 0.332 0.344). The launcher was dry-run with a fake python: phase 1 in phase1_field_on/, the
  checkpoint copied once, phase 2 in phase2_field_off/ finds it; started again, phase 1 is skipped.
- 2026-11-19 05:15:22: both runs started (2 GPUs). GPU 0: `bash how_this_was_started.sh 0 field`;
  GPU 1: `bash how_this_was_started.sh 1 ordinary`.
  config.json against the first eps = 3 self-attention run: the ordinary run differs only in
  debug.deterministic (False -> True); phase 1 in addition only in optim.iterations (20000),
  system.make_local_energy_fn (honeycomb_field.local_energy) and
  system.make_local_energy_kwargs.honeycomb_field_strength (0.5).
  Iteration 0: pmove 0.16832682 in both runs (same seed, same walkers, same weights, same moves);
  energy -312.7813 (field) and -312.667 (ordinary); the difference, -0.114, is the field term of
  those walkers.
  Speed 0.36 s per iteration on each GPU: phase 1 about 2 hours, each run about 20 hours.

## Results

After the runs.
