# psiformer_eps5 — self-attention, ε = 5

Trained with the authors' code and released configuration: 6 electrons, 9 sites,
interaction strength U = 10.623473 code units, 200,000 iterations, batch 1024, KFAC.
Started 2026-10-26 02:14:32 UTC on GPU 0, finished 2026-10-26 13:45:15 (11.5 hours,
4.8 iterations per second).

23 checkpoints; the last, `qmcjax_ckpt_199999.npz`, is the trained network.

Energy over the last 20 % of training: -59.5332 ± 0.0001 meV per electron
(drift +0.0008 ± 0.0002); the paper's Fig. 5: about -59.5.
The same network with frozen weights: `../../evaluation/psiformer_eps5/`.

What each file is: `../README.md`.
