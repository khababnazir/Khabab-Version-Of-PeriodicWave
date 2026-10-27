# Stage 8 analysis — everything after training

These files read the four trained runs in `runs/stage08/training/` and write
to `runs/stage08/evaluation/`, `runs/stage08/figures/` and
`runs/stage08/results/`. Nothing here changes a training run.

## Files, in the order they are used

| File | What it is for | Writes |
|---|---|---|
| `analyse_runs.py` | Reads the energy recorded at every training iteration (`train_stats.csv`), converts it to meV per electron, averages the last 20 % with a blocking error bar, checks for drift, and plots the training curves against the paper's Fig. 5. | `figures/1_training_curves_vs_fig5.png` |
| `evaluate_weights.py` | Evaluates **one** trained network with its weights **frozen** at the final checkpoint: 10,000 steps × 1024 walkers, the authors' network, sampler and local energy, the configuration checked against the run's `config.json`. Takes the GPU lock that training uses too, so it never shares a GPU with another job, even when started by hand. Measures the energy (split into kinetic, moiré and Coulomb) and, on the same samples, the observables of `observables.py`. | `evaluation/<run>/` |
| `launch_evaluation.sh` | Runs `evaluate_weights.py` for all four networks, one queue per GPU; a queue stops if its GPU is busy. About 12 minutes in total. A finished evaluation is never repeated. | `evaluation/logs/` |
| `observables.py` | The functions that turn sampled electron positions into the density, the pair correlation, the site-filling patterns and the structure factor S(q) — and that normalise them afterwards. No files; used by the two scripts around it. | — |
| `plot_weights.py` | Makes figures 2–6 and the two result tables from the evaluations. Refuses evaluation folders whose files disagree about how many steps they contain. | `figures/2_…` to `figures/6_…`, `results/*.csv` |
| `check_translation_symmetry.py` | Tests whether a trained network keeps the translation symmetry of the moiré lattice (it does not: moving every electron by one lattice vector changes log\|Ψ\| by about 0.5–2.9 on average). This explains the uneven site filling in figures 3 and 6. Runs on the CPU; prints only. | — |
| `test_observables.py` | Five checks of `observables.py`: the sites are the potential minima, minimum-image distances are the shortest, the three honeycombs are the three lowest-Coulomb arrangements, uncorrelated electrons give flat maps and S(q) = 1, and a perfect honeycomb gives the S(q) worked out by hand. | — |

## How to run

From this folder (`learn/stage08_first_run/analysis`):

```bash
export PATH="$HOME/.local/bin:$PATH"            # for uv
JAX_PLATFORMS=cpu ../../../.venv/bin/python test_observables.py

uv run --no-project --with matplotlib --with pandas --with pyblock python analyse_runs.py
bash launch_evaluation.sh                                         # GPUs, about 12 minutes
uv run --no-project --python ../../../.venv/bin/python --with matplotlib python plot_weights.py
../../../.venv/bin/python check_translation_symmetry.py                # CPU, under a minute
```

The project environment has no matplotlib; `uv run --no-project ... --with
matplotlib` adds it for one command without changing the environment.

## Reading order

1. `observables.py` — what is measured, and why the pair correlation shows
   order that the density alone can hide.
2. `evaluate_weights.py` — how the authors' code is used with frozen weights.
3. `plot_weights.py` — the normalisations and the figures.
4. `runs/stage08/figures/README.md` — why each figure looks the way it does.
