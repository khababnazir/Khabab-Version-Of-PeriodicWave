# psiformer_eps10 — self-attention, ε = 10, weights frozen

The final checkpoint `qmcjax_ckpt_199999.npz` (md5 b0a0444fca41d3e3017dc604505df20b) of `../../training/psiformer_eps10/`,
evaluated for 10,000 steps × 1024 walkers (10,240,000 configurations)
in 7.7 minutes (21.6 steps per second).

Energy: -32.2754 ± 0.0001 meV per electron
      = kinetic 19.749 + moiré -25.114 + Coulomb -26.910.
Spread of the local energy: 2.3 meV (whole system).
Sanity check: first 200 steps -57.36852, end of training -57.37064
(code units, total energy), passed.

Structure: S(κ) = 0.672; honeycomb fillings 1.35 % (random: 2.88 %);
site occupations from 0.23 to 0.78 (2/3 if uniform).

What each file is: `../README.md`.
