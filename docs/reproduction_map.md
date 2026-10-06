# Reproduction map — Geier, Nazaryan, Zaklama & Fu, *Self-attention neural network for solving correlated electron problems in solids*, Phys. Rev. B **112**, 045119 (2025)

DOI: 10.1103/qxc3-bkc7 · 16 pages, 5 appendices

## 1. What the paper claims

Three results, in the authors' own ordering:

1. A self-attention neural-network wave function (Psiformer-style, adapted to periodic boundary conditions) optimized by variational Monte Carlo reaches **lower ground-state energies than band-projected exact diagonalization (BP-ED)** for a moiré heterobilayer, even when BP-ED includes five bands.
2. The number of variational parameters needed for energy convergence scales as a **power law in electron number**, N*_par ≈ (400 ± 49) × N^(2.01 ± 0.05).
3. Charge density and density–density correlations from the same ansatz show a **Fermi-liquid → generalized Wigner crystal** transition as the dielectric constant is lowered.

## 2. Physical model (the only "data" involved — it is generated, not downloaded)

2D Coulomb gas in a triangular moiré potential, spin-polarized, periodic supercell:

- H = Σ_i [ −½∇²_i + V(r_i) ] + ½ Σ_{i≠j} 1/|r_i − r_j| (dimensionless units)
- V(r) = −2 V₀ Σ_{j=1..3} cos(g_j·r + φ), g_j = (4π/√3 a_M)(cos 2πj/3, sin 2πj/3)
- WSe₂/WS₂ parameters from first principles: m* = 0.35 m_e, V₀ = 15 meV, φ = π/4, a_M = 8.031 nm
- ν = 1 per moiré cell ⇒ n = 1.785 × 10¹² cm⁻²; hBN environment ε ≈ 5
- Units: a*_B = ε (m_e/m*) a₀; Ha* = (1/ε²)(m*/m_e) Ha
- All reported runs are at filling **ν = 2/3**, on 9-, 12-, 21-, 27- and 36-site supercells (⇒ 6, 8, 14, 18, 24 electrons), at ε = 10 (weak) and ε = 5 (strong)

Long-range Coulomb under PBC is handled by 2D **Ewald summation** with a neutralizing background; the Madelung term N ξ_M/2 must be included explicitly (Appendix A, Eqs. A11–A12, following Fraser *et al.* PRB 53, 1814).

## 3. Ansatz

- Periodic feature map: feature(r) = (sin G₁ᵀr, sin G₂ᵀr, cos G₁ᵀr, cos G₂ᵀr), G_iᵀL_j = 2πδ_ij — this is the one change to Psiformer that makes it periodic.
- Linear embedding → residual tanh MLP; multi-head self-attention between perceptron layers (Eqs. 18–20), normalization 1/N with N = √(d_AttnValues Σ_j exp(q_j·k_i)).
- Orbitals are complex: φ_j(r_i) = w_{2j}·h^L + i w_{2j+1}·h^L, promoted to correlated orbitals φ_j(r_i; {r_/i}) by attention; Ψ = Σ_{m=1}^{N_det} det_ij[ φ^m_j(r_i;{r_/i}) ].
- **Deliberately absent**: envelope functions, Jastrow factor, Hartree-Fock pretraining. Initialization is random. Appendix B shows a cusp-enforcing one-parameter Jastrow gives no improvement and costs 10–20 % GPU time per step.
- SlaterNet = same MLP without attention = a neural unrestricted Hartree-Fock, and the *reference* against which correlation energy is defined.

## 4. Optimization

Metropolis–Hastings sampling of |Ψ|²; energy minimization by natural-gradient descent with **KFAC** (magnitude only — phase neglected, so the quantum geometric tensor reduces to the Fisher information matrix). Table II defaults:

| | parameter | value |
|---|---|---|
| Architecture | layers / heads per layer / attention dim / perceptron dim | 3 / 6 / 16 / 64 |
| | perceptrons per layer, determinants | 1, 4 |
| Training | iterations | 1.5 × 10⁵ |
| | learning rate | η₀(1 + t/t₀)⁻¹, η₀ = 10, t₀ = 1 × 10⁵ |
| | local-energy clipping ρ | 5.0 |
| MCMC | batch size | 4096 (1024 for the Appendix-B Jastrow runs) |
| KFAC | norm constraint / damping | 1 × 10⁻³ / 1 × 10⁻³ |

η₀ = 10 is flagged by the authors as unusually high but optimal across the 2D systems they tried. Reducing KFAC damping sometimes lowered the energy but often corrupted runs.

## 5. Reported numbers to hit

27-site supercell, ν = 2/3, 18 electrons (Table I; bold = lowest):

| N_e | ε | Self-attention NN | SlaterNet (HF) | BP-ED (1 band) |
|---|---|---|---|---|
| 18 | 10 | **−32.070(7)** | −31.35(2) | −31.32443 |
| 18 | 5 | **−59.127(9)** | −58.01(3) | −57.80848 |

Other targets: correlation energy ≈ 2 % of total at 9 sites (Fig. 5, 1–5 bands of BP-ED); N*_par power law (Fig. 4); density and pair-correlation maps at ε = 10 vs 5 (Fig. 6); BP-ED MIT at ε⁻¹ ≈ 0.11, first order, with ⟨H₀⟩ discontinuity growing 0.61 → 0.87 meV and S̄(q=κ) 0.88 → 1.31 from 21 to 27 sites (Figs. 10–12).

## 6. Software and compute

- **Data availability: none.** The authors state the data are not deposited and are available on request. There is nothing to download; reproduction means re-running the calculation.
- Stated stack: DeepMind **FermiNet** (github.com/google-deepmind/ferminet) as the VMC package, plus the **Forward Laplacian** implementation *Poor Man's Forward Laplacian* (github.com/y1xiaoc/fwdlap). JAX + KFAC, GPU.
- Their runs used MIT SuperCloud / subMIT. Budget: 1.5 × 10⁵ iterations at batch 4096 for 18–24 electrons is a multi-GPU-day-scale job per data point; the convergence study in Fig. 3(b) spans ~14 network sizes × 4 system sizes.
- The local RTX 2070 SUPER is not visible to this sandbox, so any real training run needs either that GPU driven outside the sandbox or a remote GPU target.

## 7. Parameter-count check (done)

Appendix C gives N_par = 2 d_dim d_L + L(2 N_heads d_Attn d_L + 2 N_heads d_AttnValues d_L + d_L(d_L+1)) + 2 d_L N_el N_det.

Evaluating it for the Table II default (3 layers, 6 heads, d_Attn = 16, d_L = 64, N_det = 4) at N = 18 gives **95,680 parameters**, against a scaling-law saturation estimate of 1.33 × 10⁵ (band 1.01–1.73 × 10⁵ propagating both fit uncertainties). So the stated default config sits slightly *below* the fitted saturation point for the system it is said to converge — consistent to within ~30 %, but worth resolving before using the law to size networks. Two candidate causes: d_AttnValues may not equal d_Attn (the paper never gives it separately), and the "perceptrons per layer = 1" convention may add one more W^l, b^l per layer than assumed. Full table in `param_counts.csv`.

## 8. Gaps a reproduction has to fill

*(Sections 10–12 below revise this list after reading the authors' released code — most of these are now answered.)*

1. **d_AttnValues** is never specified numerically (only d_Attn = 16).
2. **Ewald parameters** — the splitting parameter η and the real/reciprocal cutoffs are not given; convergence of ξ_M must be established independently.
3. **MCMC details** — proposal step size / adaptation, number of decorrelation steps per optimization step, burn-in: not stated.
4. **Which network size was used for which reported energy** — Fig. 8 lists configs but does not map them onto system sizes; Table I energies therefore have an unstated architecture (presumably the Table II default).
5. **Cluster geometries** — 21- and 27-site k-point meshes are referenced to Wilhelm *et al.* PRB 103, 125406; the supercell vectors L₁, L₂ for each site count must be reconstructed from there.
6. **BP-ED band count** for Table I is 1; Fig. 5 uses 1–5 bands at 9 sites. Reproducing the BP-ED side is the cheaper, CPU-feasible half of the paper.
7. **Error bars** are the standard deviation of the batch-mean local energy at the end of optimization, not a blocked autocorrelation estimate — so quoted uncertainties are not directly comparable to standard VMC error analysis.

## 9. Suggested reproduction order (cheapest → most expensive)

1. Ewald/Madelung module for the 2D triangular supercell + unit conversions; validate ξ_M against an independent lattice sum.
2. Non-interacting band structure of H₀ on each cluster (plane-wave basis) — reproduces the moiré bands used by BP-ED.
3. **BP-ED**: 9-, 21-, 27-site clusters, 1–5 bands. Reproduces Table I's BP-ED column, Fig. 5 dots, and all of Appendix E (MBS degeneracies, n(k), ⟨H₀⟩ and S̄(κ) discontinuities, MIT at ε⁻¹ ≈ 0.11). CPU-feasible.
4. **SlaterNet** (no attention) at 6 and 18 electrons — smallest NN-VMC job, gives the Hartree-Fock reference and the correlation energy denominator.
5. **Self-attention NN** at 6 electrons, ε = 10 and 5 — first real test of the ansatz.
6. Scale to 18 electrons to hit Table I; then the Fig. 3(b)/Fig. 4 parameter sweep if the scaling law itself is a target.

---

# Code audit — github.com/mg607/PeriodicWave

Authors' own release (Geier & Nazaryan, MIT), Apache-2.0, a FermiNet fork. Installation verified by the authors 2025-10-09 (CPU, Apple M3) and 2025-10-17 (GPU), with full `pip freeze` pins for both in the README. 36 files; the physics lives in `periodicwave/pbc/` and the two architecture files `CustomPsiformer.py` / `SlaterNet.py`.

## 10. Entry points

| script | purpose |
|---|---|
| `periodicwave/configs/run_2deg.py` | install smoke test — 2 spin-up electrons, r_s = 10, homogeneous 2DEG |
| `periodicwave/configs/run_2deg_triangular_potential.py` | the moiré system; README states the defaults reproduce **Fig. 5** of the paper |
| `evaluate-energies.py` | plots energy vs step from `train_stats.csv` |
| `evaluate-density.py` | density and density–density correlators from the last checkpoint's MCMC walkers |

The triangular-potential script takes positional CLI args: `nspins` (`"6_0"`), `num_unit_cells`, `m*/m_e`, `1/ε`, `a_M` [nm], `V₀` [meV], `φ` [deg], `network_type` (`CustomPsiformer` | `SlaterNet`). Its built-in defaults are `[6,0]`, 9 cells, 0.35, 0.2 (ε = 5), 8.031, 15, 45°, `CustomPsiformer`. Results land in `results/2deg-CoulombMoire/{network}/el{n}_{m}_N{cells}_V..._U...`.

## 11. Gaps from §8 that the code closes

1. **`value_dim`** is a first-class option, set equal to `attn_dim = 16` in the moiré config — so my §7 assumption d_AttnValues = d_Attn was right.
2. **Ewald parameters** (`pbc/hamiltonian.py`): `gamma_factor = 2.8`, γ = (2.8/√A)², `truncation_limit = 5` in both real and reciprocal space (11 × 11 ordinal grid, sorted by |n|), attributed to the CASINO QMC manual. Madelung constant assembled exactly as Appendix A, with the neutralising background entering as −½N²·φ_S,q→0.
3. **MCMC** (moiré config): `burn_in = 300`, `steps = 30` between parameter updates, `init_width = 1.0`, `move_width = 0.2` with `move_width_updater = 'adaptive'`.
4. **Cluster geometries** — `pbc/lattices.py` hardcodes C₆-symmetric supercells for 3, 12, 21, 27 sites and falls back to the `largest_two_divisors` rectangle otherwise. Verified by running it:

| sites | M | L₁ | L₂ | area (a_M²) |
|---|---|---|---|---|
| 3 | [[1,1],[2,−1]] | (0, 1.7321) | (1.5, −0.866) | 2.5981 |
| 9 | [[3,0],[0,3]] | (3, 0) | (−1.5, 2.5981) | 7.7942 |
| 12 | [[2,2],[4,−2]] | (0, 3.4641) | (3, −1.7321) | 10.3923 |
| 21 | [[1,4],[5,−1]] | (−1.5, 4.3301) | (4.5, −0.866) | 18.1865 |
| 27 | [[3,3],[6,−3]] | (0, 5.1962) | (4.5, −2.5981) | 23.3827 |
| 36 | [[6,0],[0,6]] | (6, 0) | (−3, 5.1962) | 31.1769 |

Note 9 and 36 sites are *not* in the C₆ list — they get the diagonal fallback (which for the triangular basis is still the 3 × 3 / 6 × 6 supercell of Fig. 5).

5. **Units** — the code does *not* use the effective-Bohr-radius / effective-Hartree convention of the paper's §V. `convert_moire_scales()` measures length in a_M (set to 1.0) and energy in ħ²/(m* a_M²). Running it for the paper's material parameters:

| ε | energy_scale (meV) | V (dimensionless) | U (dimensionless) |
|---|---|---|---|
| 5 | 3.375559 | 4.443709 | 10.623473 |
| 10 | 3.375559 | 4.443709 | 5.311737 |

So the Hamiltonian actually integrated is −½Σ∇² − 2V Σ_i Σ_{n=1..3} cos(g_n·r_i + φ) + ½U Σ_{i≠j} 1/|r_i−r_j|, with U ∝ 1/ε. The two conventions are an overall rescaling of each other, but the code's is the one to reproduce against.

## 12. Discrepancies between the paper and the released code

These matter, because the repo is described as an *improved* version and its defaults are not Table II. In rough order of how much they could move an energy:

1. **Attention normalisation — corrected 2026-09-29: not a real difference.** Read from the typeset page, Eq. (18)'s normalisation is 𝒩 = √d_AttnValues · Σ_j exp(q_j·k_i): the square root covers d_AttnValues only (the extracted text had lost the radical, and this entry originally misread it as √(d_v·Σ exp)). So Eq. (18) is a standard softmax times a constant 1/√d_v, with query j compared against key i. The code's `jax.nn.dot_product_attention(q, k, v)` is a softmax of q_i·k_j/√d_k. The three differences — which index carries the query, the 1/√d_k inside the exponent, the 1/√d_v outside — are all absorbed by renaming or rescaling learned weights (W_q ↔ W_k, W_q → √d_k·W_k, W_o → W_o/√d_v). **Same function family; only the parameterisation's scale differs**, which can affect initialisation and optimisation but not what the ansatz can represent. `learn/stage05_attention/test_psiformer.py` verifies the two agree after that rescaling. (Side finding: `dot_product_attention` returns float64 for float64 inputs but is only ~1e-7 accurate against an explicit float64 softmax.) The earlier sentence calling these "different ansätze" was wrong.
2. **Learning rate.** Schedule form agrees: η(t) = rate·(1 + t/delay)^(−decay), decay = 1. Magnitudes do not — paper η₀ = 10 with t₀ = 1 × 10⁵; moiré config `rate = 0.1`, `delay = 2 × 10⁴`; `base_config` default `rate = 0.05`, `delay = 1 × 10⁴`. A factor of 100 in η₀ is the single thing I would pin down before trusting a reproduction run, and the likeliest explanation is that η₀ = 10 belongs to the tanh/no-layer-norm network of the paper.
3. **Activation and layer norm.** Paper Eqs. (9)/(13) are residual tanh with no normalisation; repo defaults are GELU plus layer norm after every attention and MLP block, with the README noting GELU/ELU "typically outperform TANH".
4. **Architecture and training budget.** Paper Table II: 3 layers, 6 heads, 1 perceptron per layer, batch 4096, 1.5 × 10⁵ iterations. Moiré config: 4 layers, 4 heads, 2 perceptrons per layer, batch 1024, 2 × 10⁵ iterations. Parameter counts from the Appendix-C formula generalised to n_perc perceptrons per layer: repo net 102,144 (N = 6) / 108,288 (N = 18); Table II net 89,536 / 95,680.
5. Matmul precision is forced to float32 (`jax_default_matmul_precision`), kinetic energy via `folx` forward Laplacian, `jastrow = "NONE"`, null envelope, `complex = True`, `clip_local_energy = 5.0`, KFAC damping and norm constraint both 1 × 10⁻³ — these all match the paper.
6. Minor: the reciprocal-space Ewald sum uses `jnp.cos` rather than a complex phase (valid because the G-set is inversion-symmetric), and `madelung_recip` contains a `- 0*(...)` dead term. Neither is a bug, but an independent check of ξ_M is still worth doing since it enters every energy as N ξ_M/2.
7. **The released kinetic-energy path drops Im E_loc from the gradient (found in Stage 6, 2026-09-30; revised 2026-10-01 after an external review).** The folx kinetic path that the moire config uses (`run_2deg_triangular_potential.py` line 137) returns `jnp.real` of the local energy. The complex branch of `loss.py` exists in two forms. With clipping on (`clip_local_energy > 0`; the moire config uses 5.0) it has the correct phase coefficient when given complex E_loc: in the limit where clipping changes no values it equals the exact estimator 2 Re<(E_loc - Re E_mean) conj(O)>, with O = d log|Psi| + i d(phase) (verified numerically to 1e-16). Clipping that does change values biases the gradient, by design, in exchange for lower variance. With clipping off it counts the Im E_loc term twice. That code is inherited unchanged from DeepMind FermiNet (c4312c3, loss.py lines 232/269/272), and the moire config does not reach it. Fed a real E_loc, both forms reduce to 2<(Re E_loc - E) d log|Psi|/dtheta>: the term 2<Im E_loc d(phase)/dtheta> is lost, and phase derivatives drop out of the gradient entirely. The paper (p. 6) says KFAC "operates only on the absolute magnitude of the wave function, while the phase is neglected", so the metric reduces to the Fisher matrix. It does not say explicitly whether the gradient is restricted the same way. In the released code it is. The energy estimator is unaffected (it averages Re E_loc, and the exact average of Im E_loc is zero for a Hermitian H): the exact energy of the trained Psi is still a variational upper bound, though a finite Monte Carlo estimate can fluctuate below it. The restriction can only change which state the optimiser reaches. Measured in `learn/stage06_natural_gradient`: in a 1-electron grid-quadrature test with a complex Psi, the real-only gradient and the clip-off branch with complex E_loc are both 69% off the exact gradient; the clip-on branch with complex E_loc (nothing clipped) agrees to 1.6e-14; with a real Psi all forms agree to 1.7e-14. In training (6 electrons, 9 sites, moire potential without Coulomb, 2,624-parameter SlaterNet with D = 1, exact SR with the full quantum geometric tensor, kfac-jax-style norm constraint, 3,000 iterations), the energy above the exact non-interacting value in these runs was, in meV (approximate errors): full gradient +0.018 +- 0.023 (seed 0) and +0.036 +- 0.025 (seed 1); real part only +3.087 +- 0.033 (seed 0) and +1.124 +- 0.023 (seed 1). Not shown: whether it matters at the authors' scale (KFAC with the Fisher metric, Coulomb, ~1e5-parameter nets, 2e5 iterations) or under other optimiser settings. Stage 8 test: rerun the Fig. 5 system with the complex E_loc kept, via either the 'default' kinetic path or the folx path with `jnp.real` removed. Check the chosen path's imaginary part for completeness first.

## 13. Revised reproduction path

The NN half no longer needs reimplementation — it needs an environment, a GPU, and a decision about which of the §12 differences to keep. Order:

1. Build the environment from the README's pinned freeze list (jax 0.7.2 CPU / 0.8.0 CUDA12, `folx` at microsoft/folx@d05c107, `kfac-jax` at the pinned commits, `pyblock`, `distrax`, `tfp-nightly`) and run `run_2deg.py` as a smoke test — the authors ran it CPU-only on an M3, so it does not need a GPU.
2. Decide the fidelity target: *paper-faithful* (set `mlp_activation_fct="TANH"`, `use_layer_norm=False`, 3 layers / 6 heads / 1 perceptron, batch 4096, and resolve η₀) versus *code-as-released* (repo defaults, expected to be as good or better but not the published numbers).
3. Reproduce Fig. 5 at 9 sites / 6 electrons, ε = 5 and 10, both `CustomPsiformer` and `SlaterNet` — the smallest run that yields a correlation energy.
4. Only then go to 27 sites / 18 electrons for Table I.
5. BP-ED is still not in this repo. Reproducing the BP-ED column of Table I, the Fig. 5 dots, and all of Appendix E means writing that independently — and it remains the CPU-feasible half.
