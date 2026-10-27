# Stage 8 — the authors' code on the paper's system (Fig. 5)

From this stage on, the paper's numbers come from the authors' own code
(PeriodicWave, in `external/PeriodicWave`, pinned at 6525d2f), run with their
released configuration: 6 spin-polarised electrons on a 3 × 3 moiré supercell
(ν = 2/3), ε = 5 and 10, self-attention network and SlaterNet, 200,000
iterations each. Everything we built in Stages 3–7 serves as a cross-check.

The **training** files are in this folder; everything done **after** training
(analysis, evaluation with frozen weights, figures) is in `analysis/`.
All data and figures are written to `runs/stage08/` (see the `README.md` there).

## Files in this folder

| File | What it is for |
|---|---|
| `run_moire.py` | Trains **one** run with the authors' code. It executes their config script and intercepts the call to `train.train`, so the configuration is exactly theirs; it changes only the output folder and the seed (fixed, so a run can be repeated), and optionally the iteration count or the Laplacian method. Safety before JAX starts: one process per folder (a lock), a resumed run must be the same experiment, and a finished run (a complete final checkpoint) exits at once. |
| `launch_fig5.sh` | Starts the four runs of Fig. 5, one queue per GPU (GPU 0: ε = 5, GPU 1: ε = 10; self-attention first, then SlaterNet). Each queue holds `runs/stage08/gpu_locks/gpuN.lock`, so two jobs never share a GPU. |
| `test_stage08.py` | The authors' local energy equals ours (Stages 4 + 7) on one wave function: the `folx` path gives the real part, the `default` path the full complex value. Without this, nothing from earlier stages could check their numbers. |
| `test_run_moire.py` | The checkpoint check that `run_moire.py` relies on before resuming or declaring a run finished (incomplete, truncated, wrongly named or damaged files are rejected; tested on small stand-in checkpoints, so no trained run is needed), and that a restart keeps the iteration target the run was started with. |
| `test_launcher.sh` | The queue logic of `launch_fig5.sh`, with stand-in programs instead of training: a busy GPU stops the queue, a failed run does not, a second queue for the same GPU refuses to start. It counts the runs started in each case and fails if a count is wrong. |
| `analysis/` | Everything after training. Its own `README.md` lists the files. |

## How to run

From `~/moire-nnvmc`, with the project environment (`source .venv/bin/activate`):

```bash
# tests (no GPU needed)
cd learn/stage08_first_run
JAX_PLATFORMS=cpu python test_stage08.py      # about a minute
JAX_PLATFORMS=cpu python test_run_moire.py    # seconds
bash test_launcher.sh                          # seconds

# training: about 21 hours for all four runs on two GPUs
bash launch_fig5.sh
tail -f ../../runs/stage08/training/logs/gpu0.log
```

A run that was stopped resumes from its last checkpoint (written every 30
minutes) when `launch_fig5.sh` is started again; a finished run exits at once.

## Reading order

1. `run_moire.py` — how the authors' configuration is captured and what we change.
2. `launch_fig5.sh` — how the four runs are queued.
3. `test_stage08.py` — why we can trust their Hamiltonian.
4. `analysis/README.md` — what was done with the trained networks.
