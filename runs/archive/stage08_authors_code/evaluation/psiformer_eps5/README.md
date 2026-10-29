# psiformer_eps5 — self-attention, ε = 5, weights frozen

The final checkpoint `qmcjax_ckpt_199999.npz` (md5 178b4f1cbd42cd7035cfcc07d837ecd3) of `../../training/psiformer_eps5/`,
evaluated for 10,000 steps × 1024 walkers (10,240,000 configurations)
in 7.8 minutes (21.4 steps per second).

Energy: -59.5319 ± 0.0002 meV per electron
      = kinetic 20.955 + moiré -25.408 + Coulomb -55.079.
Spread of the local energy: 3.5 meV (whole system).
Sanity check: first 200 steps -105.81710, end of training -105.82040
(code units, total energy), passed.

Structure: S(κ) = 0.643; honeycomb fillings 1.57 % (random: 3.06 %);
site occupations from 0.24 to 0.80 (2/3 if uniform).

What each file is: `../README.md`.
