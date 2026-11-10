# runs/stage08/18_electrons_27_sites_authors_code — the authors' own code, unchanged, as a reference

**What this is.** The authors' released program (`external/PeriodicWave`, their published commit
`6525d2f3eec53f085b55d485e1bff45d61ef1782`, no file of it changed; checked with `git status`
before the start) run exactly as they publish it, for the system of the paper's Table I and
Fig. 6: 18 electrons on 27 sites, the self-attention network, eps = 5 and eps = 10.

**No line of ours is in the training.** It is started with their script and their arguments
(spins, sites, m*/m_e, 1/eps, a_M in nm, V in meV, phi in degrees, network):

```bash
python external/PeriodicWave/periodicwave/configs/run_2deg_triangular_potential.py 18_0 27 0.35 0.2 8.031 15 45 CustomPsiformer   # eps = 5
python external/PeriodicWave/periodicwave/configs/run_2deg_triangular_potential.py 18_0 27 0.35 0.1 8.031 15 45 CustomPsiformer   # eps = 10
```

No wrapper (`run_moire.py` is not used), no changed setting, their own random seed (from the
clock, so a repeat would not give the same numbers). `how_this_was_started.sh` is everything
around it: one GPU made visible, the output kept in `logs/gpu1.log`, the lock of our own tools
on that GPU held with `flock`, and two passive monitors every 10 s (`nvidia-smi`,
`system_monitor.py`) into `logs/`.

**The settings are therefore those of their script**, as its own `config.json` in each run
folder records. They are not the paper's Table II:

| | their script (these runs) | paper's Table II (our own-code runs) |
|---|---|---|
| walkers | 1024 | 4096 |
| learning rate | 0.1 / (1 + t/20000) | 10 / (1 + t/100000) |
| iterations | 200,000 | 150,000 |
| attention network | 4 layers, 4 heads, 2 perceptrons per layer | 3 layers, 6 heads, 1 perceptron per layer |
| the same in both | GELU and layer norm, attention dimension 16, width 64, 4 determinants, clipping 5, KFAC norm constraint and damping 1e-3, 30 Metropolis steps, folx | |

Their README says the script's defaults "are set to reproduce results from Fig. 5" (6 electrons
on 9 sites). The script has no option for the entries above; using Table II's values would mean
changing six numbers in their file.

**Started 2026-11-10 02:14:38 server time (11:14 Korean time). GPU 1 only**, the two runs one
after the other (eps = 5, then eps = 10). Measured 0.344 s per iteration, so about 19 hours
each: eps = 5 until about 21:25 on 2026-11-10, eps = 10 until about 16:40 on 2026-11-11 (server
time). GPU 0 is running our paper-network run meanwhile.

**Where the results are.** Their script names its folders after the parameters:
- eps = 5: `results/2deg-CoulombMoire/CustomPsiformer/el18_0_N27_V4.44370908_45.0_U10.62347336/`
- eps = 10: `results/2deg-CoulombMoire/CustomPsiformer/el18_0_N27_V4.44370908_45.0_U5.31173668/`
  (the last number is the Coulomb strength in code units; the folder appears when that run starts)

In each: `config.json` (every setting), `train_stats.csv` (step, energy, ewmean, ewvar, pmove,
locstd; `energy` is for the whole system in code units, x 3.375559 / 18 gives meV per electron;
the file is written in blocks, so it lags by up to about 150 steps), `qmcjax_ckpt_<step>.npz`
(a checkpoint every 30 minutes: weights and the 1024 walkers), `device_info.log`.

Watch:  `tail -f runs/stage08/18_electrons_27_sites_authors_code/logs/gpu1.log`

**Afterwards** (analysis only; the runs stay theirs): the energy of the last 20 % of the
iterations against Table I, and S(kappa), the sublattices and the density from the walkers in
the checkpoints -- is there a honeycomb at eps = 5?

## Results

After the runs.
