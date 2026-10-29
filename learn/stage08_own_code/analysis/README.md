# Stage 8 analysis (our code) — everything after training

The same analysis as `learn/archive/stage08_authors_code/analysis/`, for the
four runs trained with our code (`../train_moire.py`). The files have the
same names as there, so each can be read next to its counterpart. They read
`runs/stage08/training/` and write to `runs/stage08/evaluation/`,
`runs/stage08/figures/` and `runs/stage08/results/`. Nothing here changes a
training run.

What differs from the archived files:

| File | Compared with the archived file |
|---|---|
| `evaluate_weights.py` | **Rewritten with our code**: network, local energy and Metropolis moves from `train_moire.py` (Stages 3–7), weights from our checkpoints. Same settings, sanity check and output files. |
| `test_evaluate_weights.py` | New: checks of `evaluate_weights.py`. |
| `observables.py` | Unchanged copy (same md5). |
| `test_observables.py` | The same five checks, with our supercell (Stage 3) and moiré potential (Stage 4) instead of the authors' functions. |
| `analyse_runs.py` | Reads our `settings.json` and our single `train_stats.csv` per run; networks named `attention` and `slaternet`. |
| `plot_weights.py` | Our folders and run names; network and ε from `summary.json`. Adds the comparison with the reference run: `own_vs_authors.csv` and figure 7. |
| `check_translation_symmetry.py` | Our networks and checkpoints. |
| `launch_evaluation.sh` | All four evaluations one after the other on **one** GPU, with a GPU record every 10 s (utilization, memory, temperature, power). |

## Files, in the order they are used

| File | What it is for | Writes |
|---|---|---|
| `analyse_runs.py` | Reads the energy of every training iteration, converts it to meV per electron, averages the last 20 % with a blocking error bar, checks for drift, and plots the training curves against the paper's Fig. 5. | `figures/1_training_curves_vs_fig5.png` |
| `evaluate_weights.py` | Evaluates **one** trained network with its weights **frozen** at the final checkpoint: 10,000 steps × 1024 walkers. Measures the energy (split into kinetic, moiré and Coulomb) and, on the same samples, the observables of `observables.py`. Takes the GPU lock that training uses too. | `evaluation/<run>/` |
| `launch_evaluation.sh` | Runs `evaluate_weights.py` for all four networks on one GPU. A finished evaluation is never repeated. | `evaluation/logs/` |
| `observables.py` | Turns sampled electron positions into the density, the pair correlation, the site-filling patterns and the structure factor S(q), and normalises them afterwards. | — |
| `plot_weights.py` | Figures 2–7 and the three result tables. Refuses evaluation folders whose files disagree about how many steps they contain. | `figures/2_…` to `figures/7_…`, `results/*.csv` |
| `check_translation_symmetry.py` | Does a trained network keep the translation symmetry of the moiré lattice? Moves every electron by one lattice vector and prints how much log\|Ψ\| changes. CPU; prints only. | — |
| `test_evaluate_weights.py` | Five checks of the evaluation: the moiré and Coulomb parts are the potentials inside the local energy; the supercell is the archived one; the checkpoint's own weights are used (and the other network's refused); a tiny evaluation writes consistent files and is recognised as finished; the sanity check stops a wrong wave function. | — |
| `test_observables.py` | Five checks of `observables.py`: the sites are the potential minima, minimum-image distances are the shortest, the three honeycombs are the three lowest-Coulomb arrangements, uncorrelated electrons give flat maps and S(q) = 1, and a perfect honeycomb gives the S(q) worked out by hand. | — |

## How to run

From this folder (`learn/stage08_own_code/analysis`), in a terminal:

```bash
export PATH="$HOME/.local/bin:$PATH"            # for uv
../../../.venv/bin/python test_observables.py         # CPU, under a minute
../../../.venv/bin/python test_evaluate_weights.py    # CPU, about a minute

uv run --no-project --with matplotlib --with pandas --with pyblock python analyse_runs.py
bash launch_evaluation.sh 0                            # GPU 0
uv run --no-project --python ../../../.venv/bin/python --with matplotlib python plot_weights.py
../../../.venv/bin/python check_translation_symmetry.py   # CPU, a few minutes
```

The project environment has no matplotlib; `uv run --no-project ... --with
matplotlib` adds it for one command without changing the environment.

## Reading order

1. `evaluate_weights.py`, next to the archived one — what changes when our
   code replaces the authors'.
2. `plot_weights.py` — only the comparison at the end is new.
3. `runs/stage08/figures/README.md` — what each figure shows.
