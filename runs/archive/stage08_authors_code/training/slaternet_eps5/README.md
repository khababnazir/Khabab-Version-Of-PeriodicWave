# slaternet_eps5 — SlaterNet, ε = 5

Trained with the authors' code and released configuration: 6 electrons, 9 sites,
interaction strength U = 10.623473 code units, 200,000 iterations, batch 1024, KFAC.
Started 2026-10-26 13:45:15 UTC on GPU 0, finished 2026-10-26 22:56:07 (9.2 hours,
6.1 iterations per second).

19 checkpoints; the last, `qmcjax_ckpt_199999.npz`, is the trained network.

Energy over the last 20 % of training: -57.9244 ± 0.0010 meV per electron
(drift -0.0007 ± 0.0018); the paper's Fig. 5: about -57.9.
The same network with frozen weights: `../../evaluation/slaternet_eps5/`.

What each file is: `../README.md`.
