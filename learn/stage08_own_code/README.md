# Stage 8 — the paper's Fig. 5 and Table I with our own code

Stage 8 trains the self-attention network and SlaterNet at ε = 5 and 10 with
the code we wrote in Stages 3–7, for two systems:

| System | Paper | Data (weights, evaluations, figures, tables) | Analysis scripts |
|---|---|---|---|
| 6 electrons on 9 sites | Fig. 5 | `runs/stage08/6_electrons_9_sites/` | `analysis/6_electrons_9_sites/` |
| 18 electrons on 27 sites | Table I, Fig. 6 | `runs/stage08/18_electrons_27_sites/` | `analysis/18_electrons_27_sites/` |

The training program (`train_moire.py`) and the frozen-weight evaluation
(`analysis/evaluate_weights.py`) serve both systems. The code parts:

| Part | Our code |
|---|---|
| wave functions | `stage03_slater/slaternet.py`, `stage05_attention/psiformer.py` (their "released settings" sections) |
| sampling | `stage04_mcmc/metropolis.py` |
| kinetic energy and moiré potential | `stage04_mcmc/local_energy.py` |
| Coulomb energy | `stage07_ewald/ewald.py` |
| energy gradient | `stage06_natural_gradient/gradient.py` |
| optimiser | KFAC from the `kfac-jax` library, with the authors' settings |

The settings (network sizes, activation, layer norm, KFAC, learning rate,
batch, Metropolis steps) are those of the authors' released code, so that
our runs can be compared with the reference run made with their code. That
run is archived in `learn/archive/stage08_authors_code/`, its data in
`runs/archive/stage08_authors_code/`. The 18-electron runs use the paper's
Table II settings instead (`train_moire.py --settings table2`).

## Status (2026-11-02)

1. **Networks** — the released settings added to Stages 3 and 5;
   `test_networks.py` checks they are the authors' networks. Done.
2. **Training program** — `train_moire.py`, connecting Stages 3–7 with KFAC;
   `test_train_moire.py` checks it. Done.
3. **Short runs on the GPUs** — 400 iterations, stopped, restarted to 800,
   for both networks: about 41 iterations/s (SlaterNet) and 12 (attention
   network) on an RTX 3090. Done.
4. **The four runs** of Fig. 5 (`launch_fig5.sh`), compared with the
   reference; SlaterNet first on each GPU. Started 2026-10-28 09:02. All
   four finished (200,000 iterations); the results are in
   `runs/stage08/6_electrons_9_sites/training/README.md`. `attention_6e_eps5` was interrupted once:
   GPU 1 dropped off the PCIe bus at iteration 190,863 (2026-10-28 16:00), and
   after a reboot the run continued from `checkpoint_174540.npz` on GPU 0. Done.
5. **Frozen-weight evaluation and figures** (`analysis/`): the four final
   wave functions evaluated with their weights fixed, the same figures and
   tables as for the reference run, and a comparison with it. Frozen-weight
   energies within 0.026 meV per electron of the reference run; no
   honeycomb at 9 sites, as for the reference run. Results in
   `runs/stage08/6_electrons_9_sites/results/`, figures in its `figures/`. Done.
6. **Gradient test** (2026-11-02): SlaterNet eps = 10 at 18 electrons, continued
   from its checkpoint 33,545 (before its energy rose) twice, with the
   authors' gradient and with the full one (imaginary part of E_loc kept),
   everything else unchanged (`branch_run.py`, `launch_gradient_test.sh`).
   Data in `runs/stage08/18_electrons_27_sites/gradient_test/`.
7. **18 electrons on 27 sites** (the paper's Table I). Our first attempt
   was slower than the authors' code and did not fit 4096 walkers on a GPU;
   `speed_study/speed_study_report.pdf` explains why. Fixed by computing the
   kinetic energy with folx and adding the determinants with slogdet (both
   added next to the Stage 3 and 4 versions, which stay the default), and
   `train_moire.py --electrons 18 --settings table2`. Started 2026-10-30,
   finished 2026-11-01, evaluated with frozen weights 2026-11-02; results in
   `runs/stage08/18_electrons_27_sites/README.md`. Done.

## Files in this folder

| File | What it is for |
|---|---|
| `train_moire.py` | Trains **one** network at one ε with our code: Metropolis moves (Stage 4), local energy (Stages 4 + 7), clipped gradient (Stage 6), KFAC step. All settings are the reference run's. Continues from its newest checkpoint when started again. |
| `launch_fig5.sh` | **6 electrons, 9 sites.** Starts the four runs, one queue per GPU (GPU 0: ε = 10, GPU 1: ε = 5; SlaterNet first, then the attention network). Each queue records its GPU every 10 s (utilization, memory, temperature, power), and CPU and memory (`system_monitor.py`). Data in `runs/stage08/6_electrons_9_sites/training/`. |
| `launch_table1.sh` | **18 electrons, 27 sites.** The runs of Table I with the paper's Table II settings, one queue per GPU (GPU 0: ε = 5, GPU 1: ε = 10; attention network first, then SlaterNet). `launch_table1.sh test` first runs the 2,000-iteration learning-rate check. Same records as `launch_fig5.sh`. Data in `runs/stage08/18_electrons_27_sites/training/`. |
| `branch_run.py` | Makes a new training folder from **one** checkpoint of an existing run, with only the gradient convention changed (`make`), and compares two runs row by row (`compare`). The original folder is only read. |
| `launch_gradient_test.sh` | **18 electrons, 27 sites.** The gradient test: SlaterNet eps = 10 from checkpoint 33,545 with the authors' gradient (GPU 1) and the full one (GPU 0), to 76,070 iterations, each then compared with the original and evaluated with frozen weights. `launch_gradient_test.sh check` first runs 20 iterations of each, to check that the saved state comes back. Same records as `launch_fig5.sh`. Data in `runs/stage08/18_electrons_27_sites/gradient_test/`. |
| `system_monitor.py` | Every 10 s: CPU and memory of the machine and of one job, to a CSV beside the GPU record. Started by every launcher. |
| `speed_study/` | Why 18 electrons were slow, and the fix: report, measurements, scripts. |
| `test_train_moire.py` | The settings equal the reference run's; KFAC receives Stage 6's gradient; one complete update (local energies, gradient, KFAC step) from the reference run's trained weights equals one of the authors' code; a restart continues exactly; a branch continues like the original; a damaged or incomplete checkpoint never counts as finished, nor is it trained over from zero. |
| `test_networks.py` | Our networks with the released settings against the authors' networks: same settings, parameters, initial distribution, function and derivatives with their trained weights, local energy, and the same layers for KFAC. |
| `analysis/` | Everything after training. Shared: the frozen-weight evaluation and its observables. Then one subfolder per system with its figures and tables: `6_electrons_9_sites/` (figures 1–7, the archived analysis for our runs) and `18_electrons_27_sites/` (Table I, Fig. 6). See `analysis/README.md`. |

## Run

From this folder, with the project's environment:

```bash
../../.venv/bin/python test_networks.py       # CPU, a few minutes
../../.venv/bin/python test_train_moire.py    # CPU, a few minutes

# one run per GPU:
CUDA_VISIBLE_DEVICES=0 ../../.venv/bin/python train_moire.py --network slaternet --eps 10 \
    --out ../../runs/stage08/6_electrons_9_sites/training/slaternet_6e_eps10

# 18 electrons (Table I), from ~/moire-nnvmc, both GPUs, one run on each:
bash learn/stage08_own_code/launch_table1.sh test    # first: learning-rate check, about 30 minutes
bash learn/stage08_own_code/launch_table1.sh         # then: the four runs, about 2 days
```
