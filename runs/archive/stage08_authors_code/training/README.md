# training — the four training runs of Fig. 5

One folder per run, written by the authors' code through
`learn/archive/stage08_authors_code/run_moire.py`, plus `logs/` from `launch_fig5.sh`.
Each run folder has a short `README.md` with its own numbers.

## What is in a run folder

| File | What it is |
|---|---|
| `train_stats.csv` | **The energy recorded at every training iteration** — 200,000 rows. Columns: `step`; `energy`, the batch mean of the local energy of all 6 electrons in code units (× 3.375559 / 6 = meV per electron); `ewmean`, `ewvar`, an exponentially weighted mean and variance of it; `pmove`, the fraction of walker moves accepted; `locstd`, the spread of the local energy over the 1024 walkers. The authors' code writes this file and expects its name, so it keeps it. |
| `qmcjax_ckpt_NNNNNN.npz` | Checkpoints, one every 30 minutes and one at the last iteration (`199999`). Each holds the iteration `t`, the walkers' positions (`data`), the network weights (`params`), the optimiser state (`opt_state`) and the move width (`mcmc_width`). The last one is what `analysis/evaluate_weights.py` evaluates; the earlier ones are kept (1.7 MB each) for looking at how the wave function developed. |
| `config.json` | The full configuration the run was trained with, written at its start. The evaluation rebuilds the configuration and refuses to run if it differs from this file. |
| `overrides.json` | The authors' command-line arguments and the few settings we changed (output folder, fixed seed). |
| `starts.log`, `.training.lock` | Only in the SlaterNet runs, which were started after these were added to `run_moire.py`: one line per start, and the lock that keeps a second process out. |

## logs/

`gpu0.log` (ε = 5 queue) and `gpu1.log` (ε = 10 queue): when each run started
and finished, and anything the runs printed. The `===` lines give the times:
self-attention about 11.5 hours per run (about 4.8 iterations per second
with both GPUs busy), SlaterNet about 9.2 hours (about 6 per second).
