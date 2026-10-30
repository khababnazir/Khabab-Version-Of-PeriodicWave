# Stage 8 speed study: why 18 electrons were slow

Read **speed_study_report.pdf** first. It explains, for a newcomer, why our 18-electron
training took 0.577 s per iteration (authors' code: 0.33 s) and did not fit 4096 walkers on
one GPU, and why adding the determinants with `slogdet` and computing the Laplacian with
`folx` fixes it (0.224 s at 1024 walkers, 0.79 s and 4.87 GiB at 4096).

Everything here ran on **GPU 0 only** (one RTX 3090); GPU 1 stayed idle. Nothing in this
folder changes the training code. The code changes come in a separate commit.

## Files

| file | what it is |
|---|---|
| speed_study_report.pdf | the report (9 pages) |
| build_report.py | rebuilds the PDF from the figures and the two result files |
| make_figures.py | draws fig*.png from speed_results.jsonl |
| fig*.png | the five figures of the report |
| run_all.sh | runs every measurement below, one at a time, on GPU 0, with a GPU usage log (~16 min) |
| pieces.py | time and peak memory of one piece of an iteration (`N_EL=6` or `18`; `DETERMINANT=slogdet` to swap Stage 3's det for slogdet) |
| det_alone.py | the determinant step alone, four ways (folx or full Hessian, det or slogdet) |
| full_iteration.py | one whole training iteration (moves + folx local energy + KFAC) |
| folx_kinetic.py | Stage 4's kinetic-energy formula with folx's forward Laplacian (test copy; the real one goes into Stage 4) |
| singular_example.py | what det and slogdet give when one determinant is exactly zero (CPU) |
| jaxpr_probe.py | prints what JAX hands to folx for det and for slogdet (CPU) |
| speed_results.jsonl | one JSON line per run of run_all.sh (30 October 2026) |
| singular_example.json | output of singular_example.py, float64 and float32 |
| gpu_usage.csv | nvidia-smi every 2 s, both GPUs, during run_all.sh |
| system_usage.csv | CPU and memory every 2 s (../system_monitor.py); written by run_all.sh from now on, so not for the 30 October run |
| production_check.py | the fast and the reference local energy in a run's precision (single, GPU), Table II network at 18 electrons |
| progress.txt | start and end time of every run |
| logs/ | full output of every run |
| earlier/ | the earlier 18-electron measurements quoted in the report (pieces18, bench18, folx checks, authors' smoke tests), with their logs |

## Repeat

```
cd ~/moire-nnvmc/learn/stage08_own_code/speed_study
./run_all.sh               # about 16 minutes, GPU 0 only
cat progress.txt           # which run is going
uv run --no-project --with matplotlib --with numpy python make_figures.py        # redraw the figures
uv run --no-project --with matplotlib --with reportlab python build_report.py    # rebuild the PDF
```

The project environment has no matplotlib or reportlab; `uv run --no-project --with ...` adds them for
one command without changing the environment (as in `../analysis/README.md`).

The scripts in `earlier/` were written in /tmp and are kept as they ran. Their paths point to
/tmp and they need editing before they can run again.
