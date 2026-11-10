# Stage 8 — the paper's Fig. 5 and Table I with our own code

Stage 8 trains the self-attention network and SlaterNet at ε = 5 and 10 with
the code we wrote in Stages 3–7, for two systems:

| System | Paper | Data (weights, evaluations, figures, tables) | Analysis scripts |
|---|---|---|---|
| 6 electrons on 9 sites | Fig. 5 | `runs/stage08/6_electrons_9_sites_real_gradient/` | `analysis/6_electrons_9_sites/` |
| 18 electrons on 27 sites | Table I, Fig. 6 | `runs/stage08/18_electrons_27_sites_real_gradient/` | `analysis/18_electrons_27_sites/` |

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
   `runs/stage08/6_electrons_9_sites_real_gradient/training/README.md`. `attention_6e_eps5` was interrupted once:
   GPU 1 dropped off the PCIe bus at iteration 190,863 (2026-10-28 16:00), and
   after a reboot the run continued from `checkpoint_174540.npz` on GPU 0. Done.
5. **Frozen-weight evaluation and figures** (`analysis/`): the four final
   wave functions evaluated with their weights fixed, the same figures and
   tables as for the reference run, and a comparison with it. Frozen-weight
   energies within 0.026 meV per electron of the reference run; no
   honeycomb at 9 sites, as for the reference run. Results in
   `runs/stage08/6_electrons_9_sites_real_gradient/results/`, figures in its `figures/`. Done.
6. **Gradient test** (2026-11-02): SlaterNet eps = 10 at 18 electrons, continued
   from its checkpoint 33,545 (before its energy rose) twice, with the
   authors' gradient and with the full one (imaginary part of E_loc kept),
   everything else unchanged (`branch_run.py`, `launch_gradient_test.sh`).
   Data in `runs/stage08/18_electrons_27_sites_real_gradient/gradient_test/`. Result: the
   real branch repeats the original's rise exactly; the full branch does not
   rise, and at 76,070 its frozen energy is 0.125 meV per electron lower;
   continued to 150,000 (2026-11-03) it ends at -31.2838(5), 0.140 below the
   original's end. Done (one branch per gradient; see that folder's README
   for the limits).
7. **18 electrons on 27 sites** (the paper's Table I). Our first attempt
   was slower than the authors' code and did not fit 4096 walkers on a GPU;
   `speed_study/speed_study_report.pdf` explains why. Fixed by computing the
   kinetic energy with folx and adding the determinants with slogdet (both
   added next to the Stage 3 and 4 versions, which stay the default), and
   `train_moire.py --electrons 18 --settings table2`. Started 2026-10-30,
   finished 2026-11-01, evaluated with frozen weights 2026-11-02; results in
   `runs/stage08/18_electrons_27_sites_real_gradient/README.md`. Done.
8. **Table I with the full gradient** (2026-11-03 to 11-05): the four runs again
   with `--gradient full`, everything else identical (`launch_table1.sh full`).
   Lower energies in all four: attention by 0.005 and 0.012, SlaterNet by 0.20
   meV per electron, which brings it level with the paper's SlaterNet (eps = 10)
   or below it (eps = 5). Results in
   `runs/stage08/18_electrons_27_sites_full_gradient/README.md`. Done.
9. **Looking for the crystal at eps = 3** (from 2026-11-06): the attention network
   at eps = 3 with the authors' gradient, everything else as in Table I
   (`launch_table1.sh eps3`). Not in the paper: at eps = 5 ours shows no
   honeycomb (S(kappa) 0.72-0.73, as at eps = 10). `analysis/18_electrons_27_sites/
   checkpoint_structure.py` measures S(kappa) at every checkpoint. In
   `runs/stage08/18_electrons_27_sites_eps3_real_gradient/`. Done (2026-11-07):
   no honeycomb, S(kappa) 0.63, lower than at eps = 5; but an uneven density
   (static modulation 3.4 against 0.3 at eps = 5; on the sites dominated by
   three directions at q = (2/3) kappa), the electrons still on their sites.
   Results in that folder's README.
10. **The paper's printed network in the trainer** (2026-11-08):
   `--network-form paper` trains the attention network as the paper prints it
   (Stage 5: tanh residual blocks, no layer norm, Eq. 18's attention, one key,
   query and value matrix per head), stored in the trainer's layout
   (`psiformer.py`, last section): kfac-jax recognised none of Stage 5's layers
   as written, and every parameter fell back to its crude diagonal block.
   Initial weights are Stage 5's (0.1 x normal); the paper gives none. So paper5
   against the released run compares two implementations, their initial weights
   included; it cannot isolate tanh against GELU.
   Tests: it is Stage 5's network (wave function and both derivative parts, 6
   and 18 electrons), Appendix C's parameter count, kfac-jax sees only dense
   layers; the trainer builds it and refuses what the paper does not have; its
   fast and reference local energies agree at 27 sites; a restart continues
   exactly. 7 mutants caught. Smoke run (300 iterations, 18 electrons, eps = 5,
   GPU 0): finite, 1.08 s per iteration, higher energies than the released
   network at the same iteration (-41.4 against -52.1 meV per electron at 299).
   `launch_table1.sh paper5 [ITERATIONS]` runs it at eps = 5 against
   `attention_18e_eps5` of the real-gradient folder: first `paper5 5000` (a look
   at 5,000 iterations, no evaluation), then `paper5 150000` continues it.
   Step 1 started 2026-11-08 06:44 UTC, in
   `runs/stage08/18_electrons_27_sites_paper_network_real_gradient/`. Reviewed
   at 5,000 iterations (2026-11-09): healthy, 1.0 meV per electron above the
   released network, and one sublattice emptying (S(kappa) 0.70 -> 0.95, its
   share 0.26 -> 0.21), where the released run's evened out. Step 2 to 150,000
   started 2026-11-09 01:09 UTC, watched by `watch_run.py`. At 84,600 iterations
   (2026-11-10): the same liquid as the released network, S(kappa) 0.71.
11. **Why our densities are uneven: an open shell** (2026-11-10,
   `analysis/18_electrons_27_sites/shell_structure.py`). 18 electrons on the 27
   allowed momenta leave one place free in a shell of six, so the liquid is
   six-fold degenerate and its mixtures have an uneven density at the wave
   vectors between the shell's members. 98-99 % of our site patterns at eps = 10,
   5 and 3 lie there (69 % by chance). Our networks are on the liquid branch at
   every eps; why they do not reach the paper's honeycomb at eps = 5 is open.
   In `runs/stage08/18_electrons_27_sites_eps3_real_gradient/README.md`.

## Files in this folder

| File | What it is for |
|---|---|
| `train_moire.py` | Trains **one** network at one ε with our code: Metropolis moves (Stage 4), local energy (Stages 4 + 7), clipped gradient (Stage 6), KFAC step. All settings are the reference run's. Continues from its newest checkpoint when started again. |
| `launch_fig5.sh` | **6 electrons, 9 sites.** Starts the four runs, one queue per GPU (GPU 0: ε = 10, GPU 1: ε = 5; SlaterNet first, then the attention network). Each queue records its GPU every 10 s (utilization, memory, temperature, power), and CPU and memory (`system_monitor.py`). Data in `runs/stage08/6_electrons_9_sites_real_gradient/training/`. |
| `launch_table1.sh` | **18 electrons, 27 sites.** The runs of Table I with the paper's Table II settings, one queue per GPU (GPU 0: ε = 5, GPU 1: ε = 10; attention network first, then SlaterNet). `launch_table1.sh test` first runs the 2,000-iteration learning-rate check; `launch_table1.sh full` the four runs with the full gradient; `launch_table1.sh eps3` the attention network at eps = 3 on GPU 0 (2026-11-06); `launch_table1.sh paper5` the paper's printed attention network at eps = 5 on GPU 0 (2026-11-08). Same records as `launch_fig5.sh`. Data in `runs/stage08/18_electrons_27_sites_real_gradient/training/`. |
| `watch_run.py` | A watchdog for one training run, on the server (2026-11-09): one line every `--every` seconds into `<system>/training/logs/watch.log` -- iteration, speed, energy, acceptance, variance, S(κ) at the newest checkpoint, the GPU -- and ALERT lines for non-finite numbers, a stall, an early stop, a non-zero exit code, the GPU at 88 °C. Reads only; never restarts anything. |
| `branch_run.py` | Makes a new training folder from **one** checkpoint of an existing run, with only the gradient convention changed (`make`), and compares two runs row by row (`compare`). The original folder is only read. |
| `launch_gradient_test.sh` | **18 electrons, 27 sites.** The gradient test: SlaterNet eps = 10 from checkpoint 33,545 with the authors' gradient (GPU 1) and the full one (GPU 0), to 76,070 iterations, each then compared with the original and evaluated with frozen weights. `launch_gradient_test.sh check` first runs 20 iterations of each, to check that the saved state comes back. Same records as `launch_fig5.sh`. Data in `runs/stage08/18_electrons_27_sites_real_gradient/gradient_test/`. |
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
    --out ../../runs/stage08/6_electrons_9_sites_real_gradient/training/slaternet_6e_eps10

# 18 electrons (Table I), from ~/moire-nnvmc, both GPUs, one run on each:
bash learn/stage08_own_code/launch_table1.sh test    # first: learning-rate check, about 30 minutes
bash learn/stage08_own_code/launch_table1.sh         # then: the four runs, about 2 days
```
