# runs/stage08/18_electrons_27_sites_authors_code_table2 — the authors' code with the paper's Table II values

**What this is.** The authors' released program (`external/PeriodicWave`, their published commit
`6525d2f3eec53f085b55d485e1bff45d61ef1782`, no file of it changed) for 18 electrons on 27 sites,
the self-attention network, **eps = 10**, with the settings of the paper's Table II instead of
their script's own. (eps = 5 with these settings: not decided.)

**What is changed: seven numbers, in a copy of their script.** `run_2deg_triangular_potential_table2.py`
in this folder is their `periodicwave/configs/run_2deg_triangular_potential.py` with these lines
changed and nothing else (`changes.diff` is the output of `diff` between the two; 214 lines each):

| line | their script | this copy (paper's Table II) |
|---|---|---|
| 148 | `cfg.batch_size = 1024` | `4096` |
| 153 | `cfg.optim.iterations = 200000` | `150000` |
| 154 | `cfg.optim.lr.rate    = 0.1` | `10.0` |
| 155 | `cfg.optim.lr.delay   = 20000` | `100000` |
| 178 | `cfg.network.CustomPsiformer.num_layers = 4` | `3` |
| 179 | `cfg.network.CustomPsiformer.num_heads  = 4` | `6` |
| 183 | `cfg.network.CustomPsiformer.num_perceptrons_per_layer = 2` | `1` |

Their file in `external/PeriodicWave` is untouched (the other reference folder,
`../18_electrons_27_sites_authors_code/`, runs it as it is). Table II's remaining entries --
attention dimension 16, perceptron dimension 64, 4 determinants, clipping 5, KFAC norm
constraint and damping 1e-3 -- are their script's values already.

**Checked before any GPU time** (2026-11-10, on the CPU, with their `train.train` held back so
that only the configuration is built): the configuration this copy builds differs from the one
their script builds in exactly those seven entries; the system is 18 electrons, U = 5.3117
(eps = 10), V = 4.4437, phi = 45 degrees; the network is their CustomPsiformer with GELU and
layer norm; the random seed is theirs (from the clock).

**Not changed, on purpose:** the network form. The paper's equations print tanh blocks without
layer norm; their script sets GELU and layer norm (their network class has both: two more
values, `mlp_activation_fct` and `use_layer_norm`, would switch it). Table II does not list
them, so they are left as their script has them.

**Not started yet.** GPU 1 fell off the bus on 2026-11-10 at 02:25:35 server time, and since then
no new program can start on either GPU until the server is rebooted
(`../18_electrons_27_sites_authors_code/README.md`). After a reboot (G = the GPU, 0 or 1):

```bash
cd ~/moire-nnvmc/runs/stage08/18_electrons_27_sites_authors_code_table2
setsid nohup flock -n ~/moire-nnvmc/runs/gpu_locks/gpuG.lock bash how_this_was_started.sh G >> logs/gpuG.log 2>&1 < /dev/null &
```

One GPU. Expected: 1.06 s per iteration (their code at 4096 walkers, measured 2026-10-30), so
about 44 hours for 150,000 iterations. `how_this_was_started.sh` is everything around the
training: one GPU made visible, the output in `logs/`, `flock` on our GPU lock, two passive
monitors, and a check that the GPU can start a program (their code would otherwise run on the
CPU without a word).

Results will be in `results/2deg-CoulombMoire/CustomPsiformer/el18_0_N27_V4.44370908_45.0_U5.31173668/`:
`config.json`, `train_stats.csv` (`energy` x 3.375559 / 18 = meV per electron),
`qmcjax_ckpt_<step>.npz` every 30 minutes.

**2026-11-11, 03:14: started, on GPU 0** (G = 0), eps = 10. The written `config.json` differs
from the unmodified run's in exactly the seven values above, plus the interaction strength
(eps = 10 here, eps = 5 there) and the output path.

**2026-11-11, 03:21:00: GPU 1 fell off the bus** (`../18_electrons_27_sites_authors_code/README.md`).
**03:30:01: this run, on GPU 0, died too**, nine minutes later, at step 701 (no checkpoint yet): `CUDA error: Failed to get module function: CUDA_ERROR_UNKNOWN`
at `train.py` line 703 (the update of the sampling step width). After the fault the driver
refuses anything new on both GPUs, and here their program asked for something new. (Our own
paper-network run had gone on for hours in the same state on 2026-11-10; it does not follow
that a run of the authors' code will.) The partial output is in `lost_gpu_2026-11-11/`.
Two things to know when reading the logs: the start script printed "ended with exit code 0"
although the run had crashed, so a finished run is recognised by its last step, not by that
line; and between 03:30 and 03:48 nobody noticed -- a row count of `train_stats.csv` was taken
for progress without looking at the file's time or the process.

**2026-11-11, 03:57: started again, on GPU 0** (no cap on GPU 0), from the beginning, after the
power cycle at 03:52 (`../18_electrons_27_sites_authors_code/README.md`).

## Results

After the run.
