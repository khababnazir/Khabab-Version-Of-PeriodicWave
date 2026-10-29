# slaternet_eps10 — SlaterNet, ε = 10

Trained with the authors' code and released configuration: 6 electrons, 9 sites,
interaction strength U = 5.311737 code units, 200,000 iterations, batch 1024, KFAC.
Started 2026-10-26 13:46:26 UTC on GPU 1, finished 2026-10-26 22:55:46 (9.2 hours,
6.1 iterations per second).

19 checkpoints; the last, `qmcjax_ckpt_199999.npz`, is the trained network.

Energy over the last 20 % of training: -31.5948 ± 0.0016 meV per electron
(drift +0.0258 ± 0.0019); the paper's Fig. 5: about -31.6.
The same network with frozen weights: `../../evaluation/slaternet_eps10/`.

What each file is: `../README.md`.
