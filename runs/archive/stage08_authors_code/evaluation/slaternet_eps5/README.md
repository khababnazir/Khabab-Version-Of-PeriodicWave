# slaternet_eps5 — SlaterNet, ε = 5, weights frozen

The final checkpoint `qmcjax_ckpt_199999.npz` (md5 8ca1d66ebe8fc52509afcd8fc3631f26) of `../../training/slaternet_eps5/`,
evaluated for 10,000 steps × 1024 walkers (10,240,000 configurations)
in 2.6 minutes (63.0 steps per second).

Energy: -57.9196 ± 0.0015 meV per electron
      = kinetic 22.222 + moiré -26.374 + Coulomb -53.767.
Spread of the local energy: 25.3 meV (whole system).
Sanity check: first 200 steps -102.97657, end of training -102.94828
(code units, total energy), passed.

Structure: S(κ) = 0.650; honeycomb fillings 1.23 % (random: 3.00 %);
site occupations from 0.25 to 0.94 (2/3 if uniform).

What each file is: `../README.md`.
