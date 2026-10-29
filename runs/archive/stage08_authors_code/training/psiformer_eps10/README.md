# psiformer_eps10 — self-attention, ε = 10

Trained with the authors' code and released configuration: 6 electrons, 9 sites,
interaction strength U = 5.311737 code units, 200,000 iterations, batch 1024, KFAC.
Started 2026-10-26 02:14:32 UTC on GPU 1, finished 2026-10-26 13:46:26 (11.5 hours,
4.8 iterations per second).

24 checkpoints; the last, `qmcjax_ckpt_199999.npz`, is the trained network.

Energy over the last 20 % of training: -32.2759 ± 0.0001 meV per electron
(drift -0.0001 ± 0.0001); the paper's Fig. 5: about -32.3.
The same network with frozen weights: `../../evaluation/psiformer_eps10/`.

What each file is: `../README.md`.
