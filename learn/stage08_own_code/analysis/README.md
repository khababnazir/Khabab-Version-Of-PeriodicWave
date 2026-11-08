# Stage 8 analysis (our code) — everything after training

Two systems were trained with our code. Each has its own folder of
analysis scripts here and its own data folder in `runs/stage08/`:

| System | Paper | Scripts | Data: weights (`training/`), `evaluation/`, `figures/`, `results/` |
|---|---|---|---|
| 6 electrons on 9 sites | Fig. 5 | `6_electrons_9_sites/` | `runs/stage08/6_electrons_9_sites_real_gradient/` |
| 18 electrons on 27 sites | Table I, Fig. 6 | `18_electrons_27_sites/` | `runs/stage08/18_electrons_27_sites_real_gradient/` |

The files directly in this folder are **shared**: the frozen-weight
evaluation and the observables it measures work for both systems (27 sites
added on 2026-11-02; the 9-site results are unchanged, and a test checks the
9-site site order). Nothing here changes a training run.

## Shared (this folder)

| File | What it is for | Writes |
|---|---|---|
| `evaluate_weights.py` | Evaluates **one** trained network with its weights **frozen** at its final checkpoint (or `--checkpoint N`). Rebuilds the network from the run's `settings.json`. Measures the energy (split into kinetic, moiré and Coulomb) and, on the same samples, the observables of `observables.py`. 6 electrons: 10,000 steps × 1024 walkers; 18 electrons: 5,000 × 4096. Takes the GPU lock that training uses too. | the `evaluation/` folder beside the run's `training/` folder |
| `observables.py` | Turns sampled electron positions into the density, the pair correlation, site occupations, filling patterns (9 sites only) and the structure factor S(q). | — |
| `launch_evaluation.sh` | `bash launch_evaluation.sh GPU SYSTEM [RUN[@N] ...]`: evaluates the runs of one system, one after the other on one GPU. A finished evaluation is never repeated. | `runs/stage08/SYSTEM/evaluation/logs/` |
| `test_evaluate_weights.py` | Checks of the evaluation, at 6 and 18 electrons: the energy split, the checkpoint's own weights, consistent files, the sanity check, the network rebuilt from `settings.json`, an earlier checkpoint; only a complete evaluation (every observable, every number finite, sanity check passed) counts as finished. | — |
| `test_observables.py` | Checks of `observables.py` in the 9- and 27-site cells: sites, minimum image, honeycombs, normalisations, S(q) of a perfect honeycomb. | — |

## 6 electrons on 9 sites (`6_electrons_9_sites/`) — the paper's Fig. 5

The archived analysis (`learn/archive/stage08_authors_code/analysis/`) for
our four runs, with the same file names. Since tag `9sites-own-code` only the
paths changed (and `check_translation_symmetry.py` uses the new names of two
functions of `evaluate_weights.py`).

| File | What it is for | Writes (in `runs/stage08/6_electrons_9_sites_real_gradient/`) |
|---|---|---|
| `analyse_runs.py` | The energy of every training iteration in meV per electron, the last 20 % averaged with a blocking error bar, a drift check, and the training curves against the paper's Fig. 5. | `figures/1_training_curves_vs_fig5.png` |
| `plot_weights.py` | Figures 2–7 and the three result tables from the frozen-weight evaluations, and the comparison with the authors'-code reference run. | `figures/2_…` to `figures/7_…`, `results/*.csv` |
| `check_translation_symmetry.py` | Does a trained network keep the moiré lattice's translation symmetry? CPU; prints only. | — |


## 18 electrons on 27 sites (`18_electrons_27_sites/`) — the paper's Table I and Fig. 6

| File | What it is for | Writes (in `runs/stage08/18_electrons_27_sites_real_gradient/`) |
|---|---|---|
| `checkpoint_structure.py` | Does a crystal form during training? At every training checkpoint of the runs given: S(κ), its connected part, S(κ) with the electrons moved onto their nearest sites, the share of the electrons on each sublattice, and the share of walkers whose emptiest sublattice holds 3 electrons or fewer. CPU; reads only complete checkpoints, so it can run while a run trains. Checks itself on ideal honeycombs and an even filling at every start. | `results/checkpoint_structure_<run>.csv` beside each run, `figures/checkpoint_structure.png` in the first run's folder |
| `compare_gradients.py` | Table I with the authors' gradient against the full gradient: energies, correlation energies, training curves of both. Run after `table1_results.py` and `table1_results.py --gradient full`. | `18_electrons_27_sites_full_gradient/results/real_vs_full.csv`, `correlation_real_vs_full.csv`, `figures/5_real_vs_full.png` |
| `eps3_results.py` | The attention network at ε = 3 (our diagnostic) against ε = 5 and 10, authors' gradient: frozen and training energies and whether the evaluation settled, S(κ) and the sublattices, the share of the density away from the sites, the static density modulation at every wave vector, the pattern of the 27 site occupations at their 27 wave vectors (no aliasing), and the paper's Fig. 6 layout with a third column. | `18_electrons_27_sites_eps3_real_gradient/results/energies.csv`, `structure.csv`, `off_site.csv`, `density_modulation.csv`, `site_pattern.csv`, `figures/1_…`, `figures/2_…` |
| `gradient_test_results.py` | The gradient test: SlaterNet eps = 10 continued from iteration 33,545 with the authors' and the full gradient, against the original run, row by row and with frozen weights; whether each evaluation had settled. | `gradient_test/results/*.csv`, `gradient_test/figures/gradient_test.png` |
| `table1_results.py` | Frozen-weight and training energies against the paper's Table I, the correlation energy, where the electrons are (S(κ), sublattice occupations), and the paper's Fig. 6 for both networks. Uses the energy unit, blocking error bar and colours of `../6_electrons_9_sites/analyse_runs.py`, unchanged. | `results/*.csv`, `figures/1_…` to `figures/4_…` |

## Compared with the archived analysis (authors' code, 6 electrons)

`learn/archive/stage08_authors_code/analysis/` has files of the same names:

| File | Compared with the archived file |
|---|---|
| `evaluate_weights.py` | **Rewritten with our code**: network, local energy and Metropolis moves from `train_moire.py` (Stages 3–7), weights from our checkpoints. Same settings, sanity check and output files. |
| `test_evaluate_weights.py` | New: checks of `evaluate_weights.py`. |
| `observables.py` | Copy of the archived file, extended on 2026-11-02 to the 27-site cell (the 9 sites keep their order, tested). |
| `test_observables.py` | The same five checks, with our supercell (Stage 3) and moiré potential (Stage 4) instead of the authors' functions, plus four for the 27-site cell. |
| `analyse_runs.py` | Reads our `settings.json` and our single `train_stats.csv` per run; networks named `attention` and `slaternet`. |
| `plot_weights.py` | Our folders and run names; network and ε from `summary.json`. Adds the comparison with the reference run: `own_vs_authors.csv` and figure 7. |
| `check_translation_symmetry.py` | Our networks and checkpoints. |
| `launch_evaluation.sh` | The runs of one system one after the other on **one** GPU, with a GPU record every 10 s (utilization, memory, temperature, power) and a CPU and memory record. |

## How to run

From this folder (`learn/stage08_own_code/analysis`), in a terminal:

```bash
export PATH="$HOME/.local/bin:$PATH"                  # for uv
../../../.venv/bin/python test_observables.py         # CPU, under a minute
../../../.venv/bin/python test_evaluate_weights.py    # CPU, about two minutes

# 6 electrons, 9 sites (one GPU)
bash launch_evaluation.sh 0 6_electrons_9_sites_real_gradient
cd 6_electrons_9_sites
uv run --no-project --with matplotlib --with pandas --with pyblock python analyse_runs.py
uv run --no-project --python ../../../../.venv/bin/python --with matplotlib python plot_weights.py
../../../../.venv/bin/python check_translation_symmetry.py   # CPU, a few minutes
cd ..

# 18 electrons, 27 sites (one queue per GPU)
bash launch_evaluation.sh 0 18_electrons_27_sites_real_gradient attention_18e_eps5 slaternet_18e_eps5
bash launch_evaluation.sh 1 18_electrons_27_sites_real_gradient attention_18e_eps10 slaternet_18e_eps10 slaternet_18e_eps10@33545
cd 18_electrons_27_sites
uv run --no-project --python ../../../../.venv/bin/python --with matplotlib --with pyblock python table1_results.py
uv run --no-project --python ../../../../.venv/bin/python --with matplotlib --with pyblock python gradient_test_results.py
uv run --no-project --python ../../../../.venv/bin/python --with matplotlib --with pyblock python table1_results.py --gradient full
uv run --no-project --python ../../../../.venv/bin/python --with matplotlib --with pyblock python compare_gradients.py
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= uv run --no-project --python ../../../../.venv/bin/python --with matplotlib --with pyblock python checkpoint_structure.py RUN [RUN ...]
uv run --no-project --python ../../../../.venv/bin/python --with matplotlib --with pyblock python eps3_results.py
```

The project environment has no matplotlib; `uv run --no-project ... --with
matplotlib` adds it for one command without changing the environment.

## Reading order

1. `evaluate_weights.py`, next to the archived one — what changes when our
   code replaces the authors'.
2. `6_electrons_9_sites/plot_weights.py` — only the comparison at the end is new.
3. `runs/stage08/6_electrons_9_sites_real_gradient/figures/README.md` — what each figure shows.
4. `18_electrons_27_sites/table1_results.py` and
   `runs/stage08/18_electrons_27_sites_real_gradient/README.md` — the 27-site results.
