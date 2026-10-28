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

## Status (2026-10-28)

Step 1 of 5 is written: the networks.

1. **Networks** — the released settings added to Stages 3 and 5;
   `test_networks.py` checks they are the authors' networks.
2. Training program (`train_moire.py`), connecting Stages 3–7 with KFAC.
3. Short checks of the training: finite and sensible steps, memory, time,
   saving and resuming.
4. SlaterNet at ε = 10 first, compared with the reference; then the other three.
5. The full gradient (imaginary part of E_loc kept) as a separate experiment.

## Files in this folder

| File | What it is for |
|---|---|
| `test_networks.py` | Our networks with the released settings against the authors' networks: same settings, parameters, initial distribution, function and derivatives with their trained weights, local energy, and the same layers for KFAC. |

## Run

From this folder, with the project's environment:

```bash
../../.venv/bin/python test_networks.py      # CPU, a few minutes
```
