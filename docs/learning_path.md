# Learning path — from basic ML to reproducing Geier *et al.*, PRB 112, 045119 (2025)

**Assumed starting point:** QM and linear algebra solid, ML near zero, Python at scripting level. Adjust as we go.

**Method.** Each stage: you build a small thing from scratch (no framework), then we read the paper section it corresponds to, then we read the authors' actual code for it line by line. You answer a checkpoint question before we move on — if you can't, that's information about where to slow down, not a failure.

**Rule I'm holding myself to:** I won't write the code you're learning to write. I'll write parallel examples, narrate the reasoning, and debug what you show me.

---

## Stage 0 — The map

What the five moving parts are and how they connect (paper Fig. 2). No code. Orientation only.

**Checkpoint:** explain why minimising ⟨H⟩ over *any* family of trial functions can never give you an answer below the true ground-state energy.

## Stage 1 — A neural network, from scratch, no framework

An MLP is a function with knobs. Forward pass in NumPy by hand; then derive backprop on a two-parameter toy and code it; then fit a 1D function. Nothing about physics yet.

Read alongside: paper Eqs. (5)–(10) — these *are* an MLP, written in physics notation.
Build: `mlp_from_scratch.py`, fits sin(x) on [0, 2π].

**Checkpoint:** in Eq. (9), h^{l+1} = h^l + tanh(W h^l + b), what is the `h^l +` doing and what breaks if you remove it?

## Stage 2 — JAX

`grad`, `jit`, `vmap`. Redo Stage 1 in ~20 lines. Why the code is written in a functional style with parameters passed around explicitly instead of stored in objects.

Read: `periodicwave/network_blocks.py`.
Build: `mlp_jax.py`, same fit, one-tenth the code.

**Checkpoint:** why can't you write `self.W = ...` anywhere in this codebase?

## Stage 3 — The wave function as a network

Antisymmetry; why a determinant is the only cheap way to get it; what "generalised" Slater determinant means and why it buys expressiveness. Build a miniature SlaterNet for 2 electrons in a 1D box and check the nodal structure.

Read: `SlaterNet.py` line by line. Paper §III A–B.
Build: `slaternet_mini.py`.

**Checkpoint:** the paper says the projection vectors need not be orthogonal (after Eq. 14). Why not?

## Stage 4 — Monte Carlo

Metropolis–Hastings from scratch; sample |Ψ|² for your toy Ψ; local energy E_loc = Ψ⁻¹HΨ and why its variance vanishes in an eigenstate. This is where the paper's error bars come from.

Read: `mcmc.py`, `hamiltonian.py`. Paper §IV A.
Build: `metropolis.py` + a local-energy check against an analytic case.

**Checkpoint:** why is it fine that Ψ is unnormalised?

## Stage 5 — Attention

Keys, queries, values on a toy problem you can inspect by hand; then what "permutation equivariant" means and why the ansatz needs it. Then `CustomPsiformer.py` line by line — including the Eq. (18)-vs-`dot_product_attention` difference we already found.

Read: paper §III C, Eqs. (15)–(23).
Build: `attention_from_scratch.py`.

**Checkpoint:** Eq. (21) says the correlated orbital is symmetric in the *other* electrons' coordinates. Why does the full wave function still come out antisymmetric?

## Stage 6 — Optimisation

Why plain gradient descent is bad here; what the quantum geometric tensor is; natural gradient; Fisher information; KFAC as a block approximation to an intractable inverse.

Read: `loss.py`, `curvature_tags_and_blocks.py`. Paper §IV B.

**Checkpoint:** the paper notes KFAC operates only on |Ψ| and drops the phase. What is thereby approximated away?

## Stage 7 — The physics setup

Periodic boundary conditions and the periodic feature map; Ewald summation and why a bare 1/r sum diverges; the Madelung term; the moiré potential; the unit conversion.

Read: `pbc/feature_layer.py`, `pbc/hamiltonian.py`, `pbc/lattices.py`. Paper Appendix A.
Build: an independent 2D Madelung-constant calculation to check theirs.

**Checkpoint:** why does the neutralising background enter as −½N²·φ_S,q→0 and not −½N·something?

## Stage 8 — First real run

Environment from the README's pinned freeze list; the 2-electron smoke test; then 6 electrons on 9 sites at ε = 5 and ε = 10, both `CustomPsiformer` and `SlaterNet`. Reproduces **Fig. 5** and gives you a correlation energy.

**Checkpoint:** read your own learning curve and tell me whether it converged.

## Stage 9 — BP-ED from scratch

Not in their repo, so you write it: moiré band structure in a plane-wave basis, band projection, Coulomb matrix elements, sparse diagonalisation. CPU-only, and the stage that tests the physics hardest. Reproduces the **BP-ED column of Table I**, the Fig. 5 dots, and all of **Appendix E** (many-body spectra, n(k), the first-order MIT at ε⁻¹ ≈ 0.11).

## Stage 10 — Scale up

27 sites / 18 electrons for **Table I**; densities and pair correlations for **Fig. 6**; the parameter sweep for **Figs. 3–4** and the N² scaling law if you want it. Needs a GPU.

---

## Where the compute lands

Stages 1–7 and 9 run on CPU, locally, in this sandbox. Stage 8 at 6 electrons is borderline-feasible on CPU (the authors verified their smoke test on a laptop). Stage 10 needs a GPU — either the machine's own, driven outside the sandbox, or a remote target added under Compute.
