# Stage 8 — the paper's Fig. 5 with our own code

Stage 8 trains the four networks of the paper's Fig. 5 (6 electrons on 9
sites, ε = 5 and 10, self-attention network and SlaterNet) with the code we
wrote in Stages 3–7:

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
`runs/archive/stage08_authors_code/`.

## Status (2026-10-29)

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
   `runs/stage08/training/README.md`. `attention_eps5` was interrupted once:
   GPU 1 dropped off the PCIe bus at iteration 190,863 (2026-10-28 16:00), and
   after a reboot the run continued from `checkpoint_174540.npz` on GPU 0. Done.
5. The full gradient (imaginary part of E_loc kept) as a separate experiment.

## Files in this folder

| File | What it is for |
|---|---|
| `train_moire.py` | Trains **one** network at one ε with our code: Metropolis moves (Stage 4), local energy (Stages 4 + 7), clipped gradient (Stage 6), KFAC step. All settings are the reference run's. Continues from its newest checkpoint when started again. |
| `launch_fig5.sh` | Starts the four runs, one queue per GPU (GPU 0: ε = 10, GPU 1: ε = 5; SlaterNet first, then the attention network). Data in `runs/stage08/training/`. |
| `test_train_moire.py` | The settings equal the reference run's; KFAC receives Stage 6's gradient; one complete update (local energies, gradient, KFAC step) from the reference run's trained weights equals one of the authors' code; a restart continues exactly. |
| `test_networks.py` | Our networks with the released settings against the authors' networks: same settings, parameters, initial distribution, function and derivatives with their trained weights, local energy, and the same layers for KFAC. |

## Run

From this folder, with the project's environment:

```bash
../../.venv/bin/python test_networks.py       # CPU, a few minutes
../../.venv/bin/python test_train_moire.py    # CPU, a few minutes

# one run per GPU:
CUDA_VISIBLE_DEVICES=0 ../../.venv/bin/python train_moire.py --network slaternet --eps 10 \
    --out ../../runs/stage08/training/slaternet_eps10
```
