# 18 electrons, 27 sites — training: the four runs of the paper's Table I, with our code

| Run | Network | ε | GPU |
|---|---|---|---|
| `attention_18e_eps5` | self-attention (Table II sizes: 3 layers, 6 heads, 1 perceptron, 4 determinants) | 5 | 0, first |
| `slaternet_18e_eps5` | SlaterNet (released sizes, 1 determinant) | 5 | 0, second |
| `attention_18e_eps10` | self-attention (Table II sizes) | 10 | 1, first |
| `slaternet_18e_eps10` | SlaterNet (released sizes) | 10 | 1, second |

All four: **18 electrons on 27 sites**, the paper's Table II settings (4096 walkers,
150,000 iterations, learning rate 10 / (1 + t/100000)), local energy from folx + slogdet,
the authors' gradient convention. Started 2026-10-30 with `learn/stage08_own_code/launch_table1.sh`,
finished 2026-11-01, no restarts. Results: `../README.md`.

## In each run folder

| File | What it is |
|---|---|
| `settings.json` | What the run is: network, ε, `"electrons": 18`, seed, iterations, and every training setting. A restart checks it. |
| `train_stats.csv` | One row per iteration (150,000). `energy`: mean real part of the local energy over the 4096 walkers, for the **whole system**, in code units — **× 3.375559 / 18 = meV per electron**. `variance` (code units²), `energy_imag`, `pmove` (fraction of moves accepted), `step_width`, `seconds` (wall time since the run last started), `learning_rate`, `fisher_step_sq` (KFAC's η² gᵀF⁻¹g; the step is capped at 1e-3). |
| `checkpoint_NNNNNN.npz` | The **model weights** after NNNNNN iterations, with the walkers, the step width and the optimiser state; one every 30 minutes. `checkpoint_150000.npz` is the final one. |
| `starts.log` | One line per start: date, GPU, target, jax / kfac_jax / folx versions. |
| `.training.lock` | Empty; held while the run trains. Not in git. |

## logs/

`gpu0.log`, `gpu1.log`: what each GPU queue printed (a line every 1,000 iterations).
`gpuN_monitor_<date>_<time>.csv`: the GPU every 10 s; `gpuN_system_<date>_<time>.csv`: CPU and
memory. The files stamped `2026-10-30_084846` are from the learning-rate test (`../lr_test/`),
those stamped `2026-10-30_121601` from these four runs.

The run folders were named `attention_eps5` etc. until 2026-11-02; the logs, written
before, still use those names.
