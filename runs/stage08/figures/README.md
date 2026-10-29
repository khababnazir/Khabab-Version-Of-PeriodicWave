# Stage 8 figures (our code) — what each one shows

The same figures as for the reference run of the authors' code
(`runs/archive/stage08_authors_code/figures/`, whose README explains in more
detail why each kind of plot looks the way it does), now for the four runs
trained with our code, plus figure 7, which compares the two.

System: 6 spin-polarised electrons in a 3 × 3 moiré supercell (9 sites,
filling ν = 2/3), ε = 10 and ε = 5. Energies in meV **per electron**.
Figure 1 is made by `learn/stage08_own_code/analysis/analyse_runs.py` from
`../training/`; figures 2–7 by `plot_weights.py` from `../evaluation/` (the
final weights frozen, 10,000 steps × 1024 walkers = 10,240,000 sampled
configurations per run). The numbers below are in `../results/`.

---

## 1_training_curves_vs_fig5.png — the energy during training

E/N during training, smoothed; dashed lines: the paper's Fig. 5, read by eye.
All four curves settle onto the dashed lines. Mean of the last 20 %:
−32.2775(1) and −31.6258(6) at ε = 10, −59.5328(1) and −57.9120(9) at ε = 5
(self-attention, SlaterNet). The gap between the two networks, the
correlation energy, is −0.652(1) meV at ε = 10 and −1.621(1) meV at ε = 5
(Fig. 5: about −0.7 and −1.6).

## 2_frozen_weight_energies.png — frozen-weight energies and their error bars

**(a)** Every run's energy minus its Fig. 5 value: open circles the training
mean, filled circles the frozen final weights. All eight points lie inside
the ±0.05 meV band to which Fig. 5 can be read. Frozen minus training:
−0.0001 and +0.0002 meV for self-attention (ε = 10, 5), +0.0017 and +0.0181
meV for SlaterNet. The two estimates belong to different wave functions (the
final one, and the average over the last 40,000 updates), so they need not
agree within their error bars; SlaterNet, whose training wanders the most,
differs the most.

**(b)** Blocking: flat for self-attention (standard error 0.12 and 0.17 µeV),
rising for SlaterNet (1.0 and 1.6 µeV), whose energies are correlated over
tens of steps — the same picture as for the reference run.

## 3_electron_density.png — where the electrons are

Bright, triangular spots on the 9 moiré sites, as for the reference run. The
sites are filled unevenly instead of 2/3 each: 0.57–0.74 (self-attention,
ε = 10), 0.48–0.84 (self-attention, ε = 5), 0.25–0.81 (SlaterNet, ε = 10) and
0.23–0.94 (SlaterNet, ε = 5). Our self-attention networks are more even than
the reference run's (0.23–0.78 and 0.24–0.80). SlaterNet at ε = 5 shows the
same "Y" as the reference run: density reaching from three sites into one
secondary minimum of the moiré potential. `check_translation_symmetry.py`
finds that none of our four networks keeps the translation symmetry of the
Hamiltonian (moving every electron by a1, a2 or a1 + a2 changes log|Ψ| by
0.3–3.0 on average, against at most 2 × 10⁻⁴ for the control), so the uneven
filling is a property of the wave functions, as it was for the reference run.

## 4_pair_correlation.png — where the other electrons are

The exchange-correlation hole in the centre, and peaks at the lattice
vectors (mostly the density modulation of figure 3). The self-attention hole
is slightly wider than SlaterNet's (right panels). At ε = 5 both networks
show two stronger spots in one direction, a trace of their symmetry-broken
states. No alternation of strong and weak spots: no honeycomb.

## 5_site_filling_and_structure_factor.png — is there a crystal?

**(a)** Honeycomb patterns occur in 1.3 %, 1.7 % and 1.2 % of the
configurations for self-attention ε = 10, ε = 5 and SlaterNet ε = 5, less
often than at random (2.9–3.1 %). SlaterNet at ε = 10 is the exception at
3.3 %, slightly more than at random (2.7 %), almost all of it one honeycomb
(3.2 %); the reference run's SlaterNet ε = 10 had 1.1 %. Two electrons
nearest one site: 14–24 %.

**(b)** S(q) is small at small q (0.30–0.44 at the smallest |q|), peaks at g
(1.47–1.66, the moiré lattice), and at κ is 0.64–0.72 in all four runs —
below 1: no honeycomb order. The point at |q| = 4.84 above 1 is again the
symmetry-broken state at ε = 5: S = 1.33 with a static part |⟨ρ_q⟩|²/N of
0.58 for SlaterNet (the reference run: 1.33 and 0.57, the same stripe), and
S = 1.18 with a static part of 0.12 for self-attention.

So the main structural result of the reference run holds for our runs too:
at 9 sites, neither network forms the honeycomb that the paper finds at 27
sites for ε = 5.

## 6_site_occupation_during_evaluation.png — is the spatial pattern settled?

All lines are flat over the 10,000 steps, and S(κ) agrees between the first
and second half of the evaluation to 3–4 digits
(`../results/electron_structure.csv`). The uneven filling of figure 3 is
stationary, not a sampling artefact.

## 7_ours_vs_authors.png — our code against the authors' code

**(a)** Our energy minus the reference run's, for the frozen weights (filled)
and the training means (open), in meV per electron. Frozen: −0.0022(2),
−0.0082(15), −0.0007(3) and +0.0257(22) for self-attention ε = 10, SlaterNet
ε = 10, self-attention ε = 5 and SlaterNet ε = 5. Every difference is below
0.032 meV, smaller than the 0.05 meV to which Fig. 5 can be read. Most are
larger than their statistical errors: the two codes started from different
random numbers and so ended at different final wave functions, just as the
frozen and training estimates of one run differ. The self-attention runs
agree best (within 0.0022 meV), and our self-attention wave functions have a
slightly smaller spread of the local energy (2.28 and 3.16 meV against 2.31
and 3.50).

**(b)** S(κ): ours (filled) and the reference run's (open) agree to 0.022 or
better: 0.671 / 0.672, 0.725 / 0.703, 0.644 / 0.643, 0.648 / 0.650.

Our evaluation ran at about 10.7 steps/s (self-attention) and 32 steps/s
(SlaterNet), half the speed of the reference evaluation. GPU 0 was in thermal slowdown
in 194 of its 219 busy samples (mean 83.8 °C, maximum 90 °C;
`../evaluation/logs/`); whether that explains all of the difference was not
investigated.
