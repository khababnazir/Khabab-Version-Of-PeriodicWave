# Stage 07 notes

## Errors in the first version of Stage 7, and how they were corrected

Stage 7 (2D Ewald summation: the Coulomb repulsion under periodic boundaries)
was first committed as `f7d8967`. A line-by-line review the same day
(2026-10-08) found the errors below. The corrected version is `ab5639d`.

The physics code itself was right: on random configurations the corrected
Coulomb energy agrees with the first version to 4e-16. What was wrong were
claims, the demos, and gaps in the tests.

### 1. A variational comparison across system sizes (the most serious)

**Claimed:** the Stage 6 wave function gives -57.105 (eps = 5) and -31.512
(eps = 10) meV per electron on 9 sites, "just above Table I's -59.127 and
-32.070, which is where a variational energy must sit".

**Why it was wrong:** the variational principle bounds the energy of the same
Hamiltonian on the same system. Table I is for 27 sites, and the energy per
electron depends on system size, so it bounds nothing for 9 sites. Table I's
own SlaterNet at eps = 10 (-31.35) lies *above* our 9-site value.

**Correction:** compare with Fig. 5, which is our system (3 x 3 supercell,
6 electrons). Read off the plot by eye (about +-0.05 meV): SlaterNet about
-57.9 and -31.6, self-attention about -59.5 and -32.3 meV per electron. Our
determinant, never optimised with Coulomb, lies above the paper's SlaterNet
at both eps. That is expected, but it is a numerical comparison, not a bound:
every single determinant lies above the true Hartree-Fock minimum, and
neither calculation bounds the other (see section 7). Fixed in
`interacting_energy.py` (docstring and printout) and in
`docs/reproduction_map.md` item 8.

The conclusion that Table I reports energy per electron was right and stands.

### 2. Test gaps: errors that would have passed every test

Found by breaking the code on purpose:

- **U not applied to the self-image (Madelung) term.** Every test passed,
  because every test used U = 1. At eps = 5 this changes the energy of a
  random configuration from -43.15 to -2.61 code units.
- **m*/m_e inverted in the formula for U.** Every test passed, with U about
  8 times too large.
- **Either factor of 1/2 in Eq. (A11).** Caught only by the comparison with
  the authors' code, not independently, although the test file claimed the
  eta test catches "any factor of 2". It cannot: those factors contain no eta.

**Corrections:**

- New test: perfect Wigner crystals have their published energies,
  eta_0 = -1.106102587 / r_s (triangular) and -1.100244420 / r_s (square).
  These are Bonsall and Maradudin, Phys. Rev. B 15, 1959 (1977), with the
  extra digits from arXiv:2103.01622. The test covers 9 electrons on all
  sites, 3 on a sqrt3 x sqrt3 sublattice, and a square 2 x 2 grid; agreement
  is 4.3e-10 or better. It depends on neither eta nor the authors' code.
- The comparison with the authors now runs at the paper's U = 10.623473 and
  also checks U itself (10.623473362 and 5.311736681, from their
  convert_moire_scales).
- The test docstrings now say what each test can and cannot catch.

**Result:** 12 deliberate breaks, every one caught. Of the four new ones, two
had passed every test before and two had been caught only by the authors'
code.

### 3. Wrong or unsupported statements in comments

- The `ewald.py` demo said its 6 electrons sat on "a triangular sublattice:
  the generalised Wigner crystal arrangement at filling 2/3". The code used
  random positions, and at filling 2/3 the crystal is a honeycomb (paper
  Sec. V C), not a triangular lattice.
- "gamma_factor = 2.8 is roughly the value that minimises the work": never
  checked, so removed.
- "Paper values: U = 10.623473": the paper quotes eps, not U. These numbers
  come from the authors' code.
- "One test exercises every formula": the eta test only catches terms that
  contain eta.
- Docs: "Fig. 12 matches -5.923 meV" became "is consistent with".

### 4. A dimensionally meaningless number

The demo printed the Madelung constant as "-4.7409 meV". That is xi_M (in
1/a_M) times the kinetic energy unit, without U, which is not the energy of
anything. It is now printed in 1/a_M only.

### 5. A misleading baseline (introduced while fixing, caught before commit)

A first replacement demo said the honeycomb crystal "saves 34.4 meV per
electron" relative to uncorrelated, uniformly spread charges. For such charges
the pair terms average to zero (the cell average of phi is zero, checked
numerically), so that baseline is only the self-image term N xi_M / 2. That
term is a finite-supercell artefact and vanishes as the supercell grows.

The demo now compares all 84 ways of putting 6 electrons on the 9 sites. Every
site is an equivalent moire minimum, so only the Coulomb energy separates the
arrangements. The lowest Coulomb energy, -59.582 meV per electron at eps = 5,
belongs to exactly the 3 honeycombs; the next arrangement is 2.597 meV per
electron higher. This is a classical problem restricted to the sites (no
kinetic energy, no relaxation off the sites), not the quantum ground state.

### 6. Smaller points

- **Error bars:** `interacting_energy.py` added the kinetic-plus-moire error
  and the Coulomb error in quadrature, although both come from the same
  walkers (correlation -0.22). It now takes the error of the total: 0.0836
  instead of 0.0847 code units for 6 electrons, so no reported number changes.
- **NaN gradient:** `jax.grad` of the Coulomb energy was NaN. The
  electron-with-itself entry, which is infinite, was computed and then
  overwritten. `coulomb_energy` now sums distinct pairs a < b, and its
  gradient matches finite differences to 9e-9. The authors' potential has the
  same NaN gradient; VMC never needs it.
- **Unused function:** `local_energy_with_coulomb`, the function training
  will use, was never called. The demo now uses it.

### 7. Wording corrected after a second review (same day)

An external review of `ab5639d` found no error in the calculation. It
reproduced the five tests, the eta check (2.7e-15 code units, i.e. our
relative 6.6e-16), U, the gradient against finite differences (9.2e-9), and
the 84-arrangement demo. It corrected three explanations:

- **The SlaterNet comparison** was worded as if the paper's energy had to lie
  below ours. Only the true Hartree-Fock minimum bounds both determinants.
  The paper's value lying lower is an expectation, not a bound, and both
  numbers are Monte Carlo estimates.
- **The 84-arrangement demo** was called "the classical limit". It is a
  restricted problem: electrons fixed on the moire sites, Coulomb energy only,
  no kinetic energy, no relaxation off the sites. A true classical limit
  would minimise over continuous positions.
- **The eta test** was said to catch mistakes in "every term that contains
  eta". It is a consistency check: it catches mistakes that change how a term
  depends on eta, not every conceivable error (adding a constant to the pair
  potential, for instance, would pass it). The outside references, Wigner
  crystals and the authors' code, are what make the overall check convincing.

### Caught before the first commit

- **Periodicity test:** it first shifted one electron by 2 cells, which cannot
  detect missing wrapping. Walkers drift more than 5 supercells in 3000
  Metropolis steps, so wrapping is essential. The test now shifts each
  electron by up to 9 cells.
- **Total vs per electron:** the interacting demo first printed total
  energies, but Table I is per electron.

### Where the checks live

The verification script (`s7_final_checks.py`) and the deliberate-break
script (`s7_mutants.sh`) are kept in the Claude Science artifact store, not
in this repo. Every number in this note comes from their output on
2026-10-08, and the same numbers are recorded in `docs/reproduction_map.md`
items 8 and 9.
