# authors_exact_code_runs/runs/18_electrons_27_sites_eps3_released_settings — the authors' code, unchanged, at eps = 3

**What this is.** Asked for by Khabab's professor on 2026-11-17: the authors' released program
(`external/PeriodicWave`, their published commit `6525d2f3eec53f085b55d485e1bff45d61ef1782`, no
file of it changed) run exactly as they publish it, for 18 electrons on 27 sites at **eps = 3**,
with **both networks**: the self-attention network (`CustomPsiformer`) and `SlaterNet`.

**No line of ours is in the training.** It is started with their script, their settings and their
arguments (spins, sites, m*/m_e, 1/eps, a_M in nm, V in meV, phi in degrees, network):

```bash
python external/PeriodicWave/periodicwave/configs/run_2deg_triangular_potential.py 18_0 27 0.35 0.3333333333333333 8.031 15 45 CustomPsiformer
python external/PeriodicWave/periodicwave/configs/run_2deg_triangular_potential.py 18_0 27 0.35 0.3333333333333333 8.031 15 45 SlaterNet
```

`0.3333333333333333` is 1/eps = 1/3 (their script reads it with `float()`; this is the number
nearest to 1/3 a computer can store). The interaction strength becomes U = 17.70578894 (code
units), against 10.62347336 at eps = 5 and 5.31173668 at eps = 10.

The released settings, as their script sets them (nothing changed):

| | self-attention (`CustomPsiformer`) | `SlaterNet` |
|---|---|---|
| walkers | 1024 | 1024 |
| iterations | 200,000 | 200,000 |
| learning rate | 0.1 / (1 + t/20,000) | 0.1 / (1 + t/20,000) |
| network | 4 layers, 4 heads, 2 perceptrons per layer, layer norm | 4 layers, 2 perceptrons per layer, layer norm |
| determinants | 4 | 1 (their script: "equivalent to Hartree-Fock") |

Why eps = 3. The paper's first author wrote on 2026-11-16 that eps = 5 lies very close to a
first-order transition (at eps about 4.9) between the Wigner crystal and the Fermi liquid, both
local minima there; at eps about 3.3 his plot has the crystal about 0.16 meV per electron below the
liquid. At eps = 3 the crystal should therefore be the clearly lower state. (Our own code at
eps = 3, `runs/stage08/18_electrons_27_sites_eps3_real_gradient/`, gave no honeycomb.)

## How it was started

`how_this_was_started.sh GPU NETWORK` (this folder), as for the other authors'-code runs: one GPU
made visible, the output kept in `logs/`, two passive monitors every 10 s (GPU: `logs/gpuG_monitor_*.csv`;
CPU and memory: `logs/gpuG_system_*.csv`) and a check before the run that the GPU can start a program.

```bash
cd ~/moire-nnvmc/authors_exact_code_runs/runs/18_electrons_27_sites_eps3_released_settings
setsid nohup flock -n ~/moire-nnvmc/runs/gpu_locks/gpu0.lock bash how_this_was_started.sh 0 CustomPsiformer >> logs/gpu0.log 2>&1 < /dev/null &
setsid nohup flock -n ~/moire-nnvmc/runs/gpu_locks/gpu1.lock bash how_this_was_started.sh 1 SlaterNet >> logs/gpu1.log 2>&1 < /dev/null &
```

Started again with the same command, their code continues from its newest checkpoint (every 30 minutes).

Output, written by THEIR code:
`results/2deg-CoulombMoire/CustomPsiformer/el18_0_N27_V4.44370908_45.0_U17.70578894/` and
`results/2deg-CoulombMoire/SlaterNet/el18_0_N27_V4.44370908_45.0_U17.70578894/`, each with
`config.json`, `train_stats.csv` (`energy` x 3.375559 / 18 = meV per electron) and
`qmcjax_ckpt_<step>.npz` (checkpoints: weights and walkers).

## Log

**2026-11-17, 02:09:36: both started**, self-attention on GPU 0, SlaterNet on GPU 1 (250 W cap).
The `config.json` their code wrote: 1024 walkers, 200,000 iterations, learning rate 0.1, interaction
strength 17.70578894 (eps = 3); SlaterNet with 1 determinant, self-attention with 4.
Speed in the first hour: self-attention 0.383 s per iteration (200,000: about 21 hours, end about
2026-11-17 23:30 UTC), SlaterNet 0.191 s per iteration (about 10.6 hours, end about 12:45 UTC).

**2026-11-17, about 02:30 to 03:12: the server could not be reached from outside** (network only:
the runs went on; uptime since 2026-11-11, no GPU fault in the kernel log).

## Results

After the runs.
