# Stage 8 figures — what each one shows, and why it looks the way it does

System: the paper's Fig. 5 system — 6 spin-polarised electrons in a 3 × 3
moiré supercell (9 sites, filling ν = 2/3), ε = 10 (weaker Coulomb
repulsion) and ε = 5 (twice as strong). Energies are in meV **per electron**.
"Self-attention" is the paper's network; "SlaterNet" is a single Slater
determinant of learned orbitals (a Hartree–Fock-like wave function).

Figure 1 is made by `learn/archive/stage08_authors_code/analysis/analyse_runs.py` from
`../training/`; figures 2–6 by `plot_weights.py` from `../evaluation/` (the
trained networks with their weights frozen, 10,000 steps × 1024 walkers =
10,240,000 sampled configurations per run). The numbers quoted below are in
`../results/`.

---

## 1_training_curves_vs_fig5.png — the energy during training

**Shows.** E/N at every training iteration, smoothed over 1000 iterations,
for the two networks; left ε = 10, right ε = 5. Dashed lines: the paper's
Fig. 5 values, read off the plot by eye (good to about ±0.05 meV).

**Why it looks like this.**
- The steep fall in the first few thousand iterations: the networks start
  from random weights, and the learning rate starts large (0.1, decaying as
  1 / (1 + t / 20,000) with the KFAC optimiser), so early steps are big.
- The self-attention curve lies below SlaterNet by about 1.6 meV at ε = 5 and
  0.7 meV at ε = 10. This gap is the *correlation energy*: a single
  determinant cannot make electrons avoid each other beyond what the Pauli
  principle does; the self-attention network can. It is larger at ε = 5
  because the repulsion is twice as strong.
- SlaterNet is noisier, with bumps (ε = 10, around 100,000 iterations): its
  local energy fluctuates about 7 times more (figure 2), and its training
  wanders by ±0.01–0.02 meV rather than settling.
- Both flatten onto the dashed lines: the reproduction of Fig. 5.
- The y-range runs from 0.4 meV below the paper's self-attention value to
  1 meV above its SlaterNet value, so both curves fit; the first iterations
  are off-scale on purpose.

---

## 2_frozen_weight_energies.png — frozen-weight energies and their error bars

**(a) Shows.** For each run, its energy minus the Fig. 5 value. Open circles:
mean of the last 20 % of training (iterations 160,000–200,000). Filled
circles: the final weights, frozen, evaluated on fresh samples. Grey band:
±0.05 meV, how accurately Fig. 5 can be read. The error bars are smaller than
the markers.

**Why it looks like this.** All eight points lie inside the grey band: both
estimates agree with the paper to the precision the paper can be read.
Within one run, the two estimates differ by more than their tiny error bars
(most for SlaterNet ε = 10: 0.021 meV). That is expected, not an error: the
frozen value belongs to the *final* weights only, the training value
averages 40,000 slightly different sets of weights. For self-attention the
frozen value is 0.0005–0.0013 meV higher than the training mean; training
probably reports energies slightly too low while the weights still move (not
tested separately — see the docstring of `evaluate_weights.py`).

**(b) Shows.** The standard error of the frozen-weight mean, recomputed after
averaging neighbouring steps into blocks of 1, 2, 4, … steps (blocking). The
circle marks the block length used for the error bar.

**Why it looks like this.** Flat for self-attention: successive evaluation
steps are practically uncorrelated (every step makes 30 Metropolis moves per
walker), so blocking changes nothing. Rising for SlaterNet: its energies are
correlated over tens of steps (reason not investigated), and blocking
enlarges the error bar accordingly. The errors are in µeV because 10 million
configurations were averaged; SlaterNet's are about 8 times larger, because
its local energy fluctuates much more (σ = 17–25 meV against 2.3–3.5 meV)
and its steps are correlated.

---

## 3_electron_density.png — where the electrons are

**Shows.** The electron density n(r) in the supercell (the parallelogram
spanned by 3a₁ and 3a₂), in electrons per a_M², from 61 million electron
positions per run. Same colour scale in all four panels. Corresponds to the
paper's Fig. 6(a, b), which is for 27 sites.

**Why it looks like this.**
- Bright spots on the 9 moiré sites: the minima of the moiré potential
  −2V Σⱼ cos(gⱼ·r + 45°), which is −4.24 V on a site.
- The spots are triangles: the potential is threefold symmetric. Around each
  site, the three directions towards the *secondary* minima (−1.55 V, at the
  centres of one kind of site triangle) are low; the three directions towards
  the maxima (+5.80 V, at the centres of the other kind) are high.
- The spots differ in brightness — sites are filled 0.23 to 0.94 instead of
  2/3 each. The networks do not have the translation symmetry of the
  Hamiltonian (`check_translation_symmetry.py`), and figure 6 shows the
  pattern does not change during sampling, so this is a property of the
  wave functions.
- SlaterNet at ε = 5, the "Y": density reaching from three sites into one
  secondary minimum. It is the most uneven of the four (0.25–0.94).
- There is no honeycomb here, unlike the paper's Fig. 6(b) at 27 sites, where
  one site in three is empty at ε = 5. Figure 5 tests this directly.

---

## 4_pair_correlation.png — where the other electrons are

**Shows.** Left: g(Δr) — given one electron at the centre (+), where the
other electrons are, in the supercell around it. Normalised to uniformly
spread, independent electrons: white (1) means as many pairs as those would
give; the colours are centred there. Right: the same averaged over
directions, g(r), up to half the supercell size (1.5 a_M; beyond that a
circle no longer fits in the cell).

**Why it looks like this.**
- The deep blue hole in the centre: two electrons are never close — the
  Pauli principle (all spins equal) and the Coulomb repulsion.
- The red spots sit at the lattice vectors (distance 1, and √3 at two
  corners). They are mostly the density modulation of figure 3 — both
  electrons of a pair sit on sites — not by themselves a sign of
  correlation: independent electrons in this potential would give peaks too.
- The self-attention hole is slightly wider than SlaterNet's (right panels):
  the correlation that the single determinant lacks.
- SlaterNet at ε = 5 is not sixfold symmetric (two dark red spots): its
  symmetry-broken state prefers one direction.
- The paper's Fig. 6(d) (27 sites, ε = 5) shows strong and weak spots
  alternating — a honeycomb. Here the spots around the centre are alike.

---

## 5_site_filling_and_structure_factor.png — is there a crystal?

**(a) Shows.** Each electron is assigned to its nearest site; each sampled
configuration is then one of the three honeycombs (6 of 9 sites filled, the
3 empty ones forming a triangular lattice of spacing √3), another
one-electron-per-site arrangement, or one with two electrons nearest the same
site. Black tick: how often honeycombs would occur if the one-per-site
arrangements were random (3 of the 84).

**Why it looks like this.** Honeycombs occur in 1.1–1.6 % of the
configurations, *less* often than at random (2.7–3.1 %): these wave
functions are not honeycomb crystals. "Two electrons nearest one site"
(14–25 %) mostly means an electron spread between sites, towards a secondary
minimum, counted with a site that already has one; it is more common at
ε = 10, where the repulsion is weaker.

**(b) Shows.** The structure factor S(q) = ⟨|ρ_q|²⟩ / N at every wave vector
of the supercell up to twice the moiré reciprocal vector g. κ: the corner of
the moiré Brillouin zone, where a honeycomb crystal has its peak (1.5 for
electrons fixed exactly on the sites, above 1 for any honeycomb order).

**Why it looks like this.**
- Small S at small q: long-wavelength charge fluctuations cost a lot of
  Coulomb energy, so they are suppressed — a general property of systems
  with long-range repulsion.
- The peak at g (1.5–1.65): the moiré lattice itself, every electron near a
  site.
- At κ, 0.64–0.70 in all four runs, below 1: no honeycomb order, for either
  network at either ε.
- One SlaterNet ε = 5 point at |q| = 4.84 (S = 1.33): one direction only, and
  mostly static (|⟨ρ_q⟩|²/N = 0.57) — a fixed density stripe from its
  symmetry-broken state, the same one seen in figures 3 and 4.

This is the main structural result of Stage 8: at 9 sites, neither network
forms the honeycomb that the paper finds at 27 sites for ε = 5. Whether that
is the physics of 6 electrons in a 3 × 3 cell or a limit of the networks is
open; the structure factor of the exact (band-projected) ground state at 9
sites, Stage 9, will decide it.

---

## 6_site_occupation_during_evaluation.png — is the spatial pattern settled?

**Shows.** The occupation of each of the 9 sites at every evaluation step,
averaged over 100 steps; dashed line 2/3, a uniform filling.

**Why it was made.** The energy's blocking analysis (figure 2b) says nothing
about the spatial pattern: walkers can sample the energy well and still move
only slowly between, say, the three translated copies of a pattern. If they
did, the uneven filling of figure 3 would be an artefact of sampling.

**Why it looks like this.** All lines are flat over the 10,000 steps, and
S(κ) agrees between the first and second half to four digits
(`../results/electron_structure.csv`). The pattern is stationary; the spread
between the lines is the uneven filling built into the wave functions.
