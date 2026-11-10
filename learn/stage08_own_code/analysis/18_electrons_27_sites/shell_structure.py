"""
Stage 8 -- the shells of one electron in the moire potential, on the momenta our cells allow.

Why.  Our trained networks show an uneven density at eps = 10, 5 and 3 (some sites filled more
often than others), at wave vectors that are not the honeycomb's.  This script asks whether
that is a finite-size shell effect.

A periodic cell of 27 (or 9) moire sites allows 27 (or 9) crystal momenta k.  For each, one
electron in the moire potential (no interaction) has a lowest-band energy E(k), found here by
diagonalising  H = -1/2 laplacian + V_moire  in plane waves exp(i (k + G) . r).  Momenta related
by the lattice's rotations by 120 degrees and by k -> -k have the same energy: a SHELL.  The
electrons fill the shells from below, as in an atom.  If the last shell is only partly filled
(an OPEN shell), the non-interacting ground state is degenerate, and so is the interacting one
as long as it stays a liquid: its states can be labelled by the momentum left out, and any
mixture of them is an equally good ground state.  A mixture has an uneven density, at the
wave vectors q = k_i - k_j between members of the open shell -- and only there.

Output (printed, and results/shell_structure.csv in the eps = 3 folder, where
site_pattern.csv is):
  1. the shells and their filling with 18 electrons on 27 sites and 6 on 9;
  2. the wave vectors k_i - k_j of the open shell, reduced into the Brillouin zone;
  3. how much of the measured site pattern (site_pattern.csv, eps = 10, 5, 3) lies on them.

Checks at every run: E(k) = E(-k) and E(k) = E(rotated k); converged in the number of plane
waves; the non-interacting energy per electron lies below the kinetic + moire energy that the
trained networks have (they also pay attention to the repulsion).

Run, from this folder:
    uv run --no-project --python ../../../../.venv/bin/python --with matplotlib --with pyblock python shell_structure.py
"""
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))                  # observables.py
sys.path.insert(0, str(HERE.parents[1]))              # train_moire.py, which adds the stage folders
import train_moire                                    # noqa: E402
import observables                                    # noqa: E402
import slaternet                                      # noqa: E402
from local_energy import moire_reciprocal_vectors     # noqa: E402

ENERGY_UNIT_MEV = 3.375559
EPS3 = HERE.parents[3] / "runs" / "stage08" / "18_electrons_27_sites_eps3_real_gradient"
LATTICE = observables.MOIRE_LATTICE                   # columns a1, a2
B = 2 * np.pi * np.linalg.inv(LATTICE).T              # columns b1, b2: the moire reciprocal vectors
V, _, PHI = train_moire.moire_constants(5.0)          # the moire potential does not depend on eps


def lowest_band(k, cutoff=6):
    """Lowest eigenvalue of -1/2 laplacian + V_moire at crystal momentum k (code units).

    V_moire(r) = -2 V sum_j cos(g_j . r + phi) = -V sum_j (e^{i phi} e^{i g_j . r} + c.c.),
    so a plane wave G is coupled to G + g_j by -V e^{i phi} and to G - g_j by -V e^{-i phi}."""
    g = np.asarray(moire_reciprocal_vectors())                             # (3, 2)
    n = np.array(list(itertools.product(range(-cutoff, cutoff + 1), repeat=2)))
    G = n @ B.T
    index = {tuple(row): i for i, row in enumerate(n)}
    H = np.diag(0.5 * np.sum((k + G) ** 2, axis=1)).astype(complex)
    steps = np.round(np.linalg.solve(B, g.T).T).astype(int)                # g_j in units of b1, b2
    assert np.allclose(steps @ B.T, g), "the g_j must be moire reciprocal vectors"
    for i, row in enumerate(n):
        for step in steps:
            j = index.get(tuple(row + step))
            if j is not None:
                H[j, i] += -V * np.exp(1j * PHI)
                H[i, j] += -V * np.exp(-1j * PHI)
    return float(np.linalg.eigvalsh(H)[0])


def reduce_to_zone(m):
    """m (in units of b1, b2) -> the shortest equivalent wave vector, as (m1, m2) and |q|."""
    shifts = [m + np.array([i, j]) for i in (-1, 0, 1) for j in (-1, 0, 1)]
    best = min(shifts, key=lambda s: (round(float(np.linalg.norm(B @ s)), 9), tuple(np.round(s, 9))))
    return best, float(np.linalg.norm(B @ best))


def allowed_momenta(n_sites):
    """The n_sites crystal momenta of the cell, reduced into the moire Brillouin zone."""
    supercell = np.asarray(slaternet.supercell_vectors(n_sites), dtype=float)
    Bs = 2 * np.pi * np.linalg.inv(supercell).T                           # supercell reciprocal vectors
    found = {}
    for i, j in itertools.product(range(-9, 10), repeat=2):
        m = np.linalg.solve(B, Bs @ np.array([i, j]))
        key = tuple(np.round(np.mod(m, 1.0) * 27).astype(int) % 27)
        found.setdefault(key, reduce_to_zone(np.mod(m, 1.0))[0])
    assert len(found) == n_sites, f"{n_sites} sites must give {n_sites} momenta, found {len(found)}"
    return np.array(list(found.values()))


def shells(n_sites, n_electrons):
    """Shells of the n_sites momenta, lowest first, and how n_electrons fill them."""
    ms = allowed_momenta(n_sites)
    energies = np.array([lowest_band(B @ m) for m in ms])
    for m, e in zip(ms, energies):                                          # symmetry and convergence
        rotation = np.array([[-0.5, -np.sqrt(3) / 2], [np.sqrt(3) / 2, -0.5]])
        assert abs(lowest_band(-(B @ m)) - e) < 1e-9, "E(k) = E(-k)"
        assert abs(lowest_band(rotation @ (B @ m)) - e) < 1e-9, "E(k) = E(k rotated by 120 degrees)"
    assert abs(lowest_band(B @ ms[1], cutoff=8) - lowest_band(B @ ms[1], cutoff=6)) < 1e-8, "plane waves converged"
    rows, filled = [], 0
    for e in sorted(set(np.round(energies, 8))):
        members = ms[np.isclose(energies, e, atol=1e-7)]
        inside = min(len(members), n_electrons - filled)
        filled += inside
        rows.append(dict(sites=n_sites, electrons=n_electrons, q=reduce_to_zone(members[0])[1], members=len(members),
                         energy_meV=e * ENERGY_UNIT_MEV, filled=inside, open=0 < inside < len(members),
                         momenta=members))
    energy_per_electron = sum(r["energy_meV"] * r["filled"] for r in rows) / n_electrons
    return rows, energy_per_electron


def difference_shells(members):
    """|q| of every k_i - k_j (i != j) between the members of a shell, reduced into the zone."""
    return sorted({round(reduce_to_zone(a - b)[1], 4) for a in members for b in members if not np.allclose(a, b)})


def main():
    table = []
    for n_sites, n_electrons in ((27, 18), (9, 6)):
        rows, per_electron = shells(n_sites, n_electrons)
        print(f"{n_electrons} electrons on {n_sites} sites: shells of the lowest band (no interaction)")
        for r in rows:
            print(f"   |k| = {r['q']:.3f}/a_M   {r['members']} momenta   E = {r['energy_meV']:8.3f} meV   "
                  f"filled {r['filled']} of {r['members']}" + ("   <-- OPEN SHELL" if r["open"] else ""))
        width = rows[-1]["energy_meV"] - rows[0]["energy_meV"]
        print(f"   band width {width:.2f} meV; non-interacting energy {per_electron:.3f} meV per electron")
        open_shells = [r for r in rows if r["open"]]
        for r in open_shells:
            r["difference_q"] = difference_shells(r["momenta"])
            print(f"   open shell at |k| = {r['q']:.3f}: its differences k_i - k_j lie at |q| = {r['difference_q']}")
        table += [{k: v for k, v in r.items() if k != "momenta"} for r in rows]
        if n_sites == 27:
            open27, energy27 = open_shells, per_electron

    # The trained networks' kinetic + moire energy must lie above the non-interacting value.
    energies = pd.read_csv(EPS3 / "results" / "energies.csv")
    for _, row in energies.iterrows():
        one_body = row["kinetic"] + row["moire"]
        assert one_body > energy27, f"{row['run']}: kinetic + moire {one_body} below the non-interacting {energy27}"
        print(f"   check, {row['run']}: kinetic + moire {one_body:.3f} meV per electron > {energy27:.3f}")

    print("the measured site pattern (site_pattern.csv): share on the open shell's difference wave vectors")
    pattern = pd.read_csv(EPS3 / "results" / "site_pattern.csv")
    allowed = open27[0]["difference_q"] if open27 else []
    for eps, group in pattern.groupby("eps", sort=False):
        total = group["site_pattern"].sum()
        by_shell = group.groupby(group["q_bz"].round(4))["site_pattern"].sum() / total
        on = sum(share for q, share in by_shell.items() if any(abs(q - a) < 1e-3 for a in allowed))
        detail = ", ".join(f"|q| {q:.3f}: {100 * share:.1f} %" for q, share in by_shell.items() if q > 0)
        print(f"   eps {eps:g}: total {total:.3f}; on the predicted wave vectors {100 * on:.1f} %   ({detail})")
        table.append(dict(sites=27, electrons=18, eps=eps, site_pattern_total=total, share_on_predicted=on))
    pd.DataFrame(table).to_csv(EPS3 / "results" / "shell_structure.csv", index=False)


if __name__ == "__main__":
    main()
