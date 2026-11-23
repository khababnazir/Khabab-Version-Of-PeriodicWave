# authors_exact_code_runs/archive/eps3_seed42_honeycomb_field — the honeycomb-field test at eps = 3

Moved here on 2026-11-23 from `../../runs/18_electrons_27_sites_eps3_seed42/`, at Khabab's request
("delete that training with field ... or move to an archived folder"). Nothing was deleted or
changed; every file has the same md5 checksum as before the move. No script uses these files any
more, and the scripts here still name the old paths, so they do not run from here.

What it was: a run of the authors' code (released settings, eps = 3, self-attention, seed 42) that
for iterations 0 to 19,999 added a field to their Hamiltonian, E_field = 0.5 sum_i sum_j
cos(kappa_j . r_i), which raises one sublattice of moire sites by 7.6 meV; then, from that
checkpoint, 180,000 iterations of their plain seed-42 script (field off); then the final weights
frozen, field off.

Result: at iteration 19,999 the raised sublattice held 2.7 % of the electrons (a honeycomb); with
the field off it stayed at 2.7 to 3.9 % to the end. Frozen energy -96.1335(1) meV per electron,
S(kappa) 2.63, sublattice shares 0.037 / 0.481 / 0.482. The same seed WITHOUT the field (the run that
stayed in `../../runs/18_electrons_27_sites_eps3_seed42/ordinary/`) formed the same honeycomb by
itself (another sublattice empty), frozen -96.1305(2); so the field turned out not to be needed.

| here | was |
|---|---|
| `honeycomb_field/` (phase1_field_on/, phase2_field_off/ with its evaluation/, phase1_check.csv) | `runs/18_electrons_27_sites_eps3_seed42/honeycomb_field/` |
| `honeycomb_field.py`, `test_honeycomb_field.py`, `run_2deg_triangular_potential_seed42_field.py` | that folder |
| `run_folder_before_the_move/` (README.md, how_this_was_started.sh, changes.diff, results/, figures/) | that folder: the files that covered BOTH runs, as they were |
| `logs/gpu0*` | that folder's `logs/` (the field run's training) |
| `logs/evaluation_seed42_gpu0*` | `authors_exact_code_runs/logs/` (its frozen evaluation) |

Phase 2 was run with `run_2deg_triangular_potential_seed42.py`, which stays in the run folder.
