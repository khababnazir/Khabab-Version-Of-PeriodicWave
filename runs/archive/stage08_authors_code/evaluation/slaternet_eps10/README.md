# slaternet_eps10 — SlaterNet, ε = 10, weights frozen

The final checkpoint `qmcjax_ckpt_199999.npz` (md5 cf1e24878d393258a18cb0de13f1281e) of `../../training/slaternet_eps10/`,
evaluated for 10,000 steps × 1024 walkers (10,240,000 configurations)
in 2.6 minutes (63.8 steps per second).

Energy: -31.6159 ± 0.0011 meV per electron
      = kinetic 19.933 + moiré -25.589 + Coulomb -25.961.
Spread of the local energy: 16.9 meV (whole system).
Sanity check: first 200 steps -56.20278, end of training -56.15629
(code units, total energy), passed.

Structure: S(κ) = 0.703; honeycomb fillings 1.11 % (random: 2.69 %);
site occupations from 0.29 to 0.85 (2/3 if uniform).

What each file is: `../README.md`.
