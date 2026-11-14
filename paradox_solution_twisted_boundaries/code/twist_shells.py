"""
PARADOX EXPERIMENT -- which one-electron levels do 18 electrons fill, with and without a twist?

Run (numpy only, a few seconds, no GPU):   python twist_shells.py

One electron in the moire potential has energy levels E(k), one "band" of them per
crystal momentum k.  In our 27-site supercell only 27 momenta k are allowed.  With
periodic boundary conditions they are k = n1 G1 + n2 G2 (G1, G2: the supercell's
reciprocal vectors).  With a twist (T1, T2) -- the wave function picks up the phase
exp(2 pi i T_i) when an electron goes once around the supercell along L_i -- they are

        k = (n1 + T1) G1 + (n2 + T2) G2            the whole grid of momenta is shifted.

18 electrons without interaction fill the 18 lowest levels.  If level 18 and level 19
have the SAME energy, the filling is not unique: an "open shell".  That is the
situation of every periodic run (learn/.../shell_structure.py): 5 electrons in a shell
of 6 equal levels.  This script shows that the twist (1/3, 1/3) lifts the tie -- the
18 lowest levels are then separated from the 19th by a gap, a "closed shell" -- and
that it is the best of the twists that keep the momentum grid three-fold symmetric.

For each k the levels are the eigenvalues of a 169 x 169 matrix: the Hamiltonian
-1/2 laplacian + V(r) written in the plane waves exp(i (k + G) . r), G = m1 b1 + m2 b2
with |m1|, |m2| <= 6 (13 x 13 = 169 of them), exactly as in shell_structure.py:
        diagonal      1/2 |k + G|^2
        off-diagonal  -V e^{+i phi} between G and G + g_j,  -V e^{-i phi} between G and G - g_j
because V(r) = -2 V sum_j cos(g_j . r + phi).   test_twist.py checks this file against
shell_structure.py and the constants against train_moire.py.

What this does NOT show: anything about the interacting electrons.  The gap found here
(about 1.1 meV) is small against the repulsion between neighbours (about 36 meV at
eps = 5).  "Closed shell" is a statement about the starting point, not about the answer.

Output: the table on the screen and ../results/one_electron_levels.csv.
"""
import csv
import itertools
from pathlib import Path

import numpy as np

ENERGY_UNIT_MEV = 3.375559                 # one code unit of energy, in meV
V = 4.443709082419185                      # the paper's 15 meV, in code units
PHI = np.pi / 4
N_ELECTRONS = 18
A = np.array([[1.0, -0.5], [0.0, np.sqrt(3) / 2]])            # columns a1, a2: the moire lattice
B = 2 * np.pi * np.linalg.inv(A).T                            # columns b1, b2: its reciprocal vectors
MOIRE_G = np.array([B[:, 1], B[:, 0] - B[:, 1], -B[:, 0]])    # g_1, g_2, g_3 of the potential
SUPERCELL = np.array([[0.0, 4.5], [3 * np.sqrt(3), -1.5 * np.sqrt(3)]])   # columns L1 = 3a1 + 6a2, L2 = 3a1 - 3a2
SUPERCELL_G = 2 * np.pi * np.linalg.inv(SUPERCELL)            # ROWS G1, G2:  G_i . L_j = 2 pi if i = j, else 0
N_SITES = 27


def band_energies(k, cutoff=6, n_bands=2):
    """The n_bands lowest energies of one electron with crystal momentum k (code units)."""
    m = np.array(list(itertools.product(range(-cutoff, cutoff + 1), repeat=2)))
    G = m @ B.T
    index = {tuple(row): i for i, row in enumerate(m)}
    H = np.diag(0.5 * np.sum((k + G) ** 2, axis=1)).astype(complex)
    steps = np.round(np.linalg.solve(B, MOIRE_G.T).T).astype(int)          # g_j in units of b1, b2
    for i, row in enumerate(m):
        for step in steps:
            j = index.get(tuple(row + step))
            if j is not None:
                H[j, i] += -V * np.exp(1j * PHI)
                H[i, j] += -V * np.exp(-1j * PHI)
    return np.linalg.eigvalsh(H)[:n_bands]


def twist_vector(twist_1, twist_2):
    """k_twist = T1 G1 + T2 G2: the shift of every allowed momentum."""
    return twist_1 * SUPERCELL_G[0] + twist_2 * SUPERCELL_G[1]


def shortest_equivalent(k):
    """k and k + (a moire reciprocal vector) are the same crystal momentum; return the shortest."""
    fractions = np.linalg.solve(B, k)                    # k = fractions[0] b1 + fractions[1] b2
    k = B @ (fractions - np.round(fractions))            # bring it near the origin first
    copies = [k + B @ np.array(shift) for shift in itertools.product((-1, 0, 1), repeat=2)]
    return min(copies, key=lambda q: (round(float(q @ q), 9), round(float(q[0]), 9), round(float(q[1]), 9)))


def allowed_momenta(twist_1=0.0, twist_2=0.0):
    """The 27 different crystal momenta of the 27-site supercell, as an array (27, 2)."""
    found = {}
    for n1, n2 in itertools.product(range(-9, 10), repeat=2):
        k = shortest_equivalent((n1 + twist_1) * SUPERCELL_G[0] + (n2 + twist_2) * SUPERCELL_G[1])
        found[tuple(np.round(k, 8))] = k
    momenta = np.array(list(found.values()))
    assert len(momenta) == N_SITES, f"expected {N_SITES} momenta, found {len(momenta)}"
    return momenta


def levels(twist_1=0.0, twist_2=0.0, cutoff=6):
    """The 27 lowest-band levels, lowest first: (momenta (27, 2), energies (27,), lowest
    energy of the SECOND band).  Energies in code units."""
    momenta = allowed_momenta(twist_1, twist_2)
    bands = np.array([band_energies(k, cutoff) for k in momenta])
    order = np.argsort(bands[:, 0], kind="stable")
    return momenta[order], bands[order, 0], float(bands[:, 1].min())


def shells(energies, tolerance=1e-7):
    """Group equal energies: a list of [first index, how many, energy]."""
    groups = []
    for i, energy in enumerate(energies):
        if groups and abs(energy - groups[-1][2]) < tolerance:
            groups[-1][1] += 1
        else:
            groups.append([i, 1, float(energy)])
    return groups


def gap(energies, n_electrons=N_ELECTRONS):
    """Energy of level n+1 minus energy of level n: zero for an open shell."""
    return float(energies[n_electrons] - energies[n_electrons - 1])


def is_threefold_symmetric(momenta):
    """Does a rotation by 120 degrees map the set of momenta onto itself?"""
    c, s = np.cos(2 * np.pi / 3), np.sin(2 * np.pi / 3)
    rotation = np.array([[c, -s], [s, c]])
    have = {tuple(np.round(shortest_equivalent(k), 6)) for k in momenta}
    return all(tuple(np.round(shortest_equivalent(rotation @ k), 6)) in have for k in momenta)


def is_inversion_symmetric(momenta):
    """Does k -> -k map the set onto itself?  (Time reversal: needed for a REAL wave function.)"""
    have = {tuple(np.round(shortest_equivalent(k), 6)) for k in momenta}
    return all(tuple(np.round(shortest_equivalent(-k), 6)) in have for k in momenta)


def total_momentum(momenta, n_electrons=N_ELECTRONS):
    """Sum of the n lowest momenta, as the shortest equivalent crystal momentum."""
    return shortest_equivalent(momenta[:n_electrons].sum(axis=0))


def describe(twist_1, twist_2):
    momenta, energies, second_band = levels(twist_1, twist_2)
    filled = 0
    print(f"\ntwist ({twist_1:.4f}, {twist_2:.4f}) of a full turn;  k_twist = {np.round(twist_vector(twist_1, twist_2), 4)}")
    print("  levels   energy (meV)   filled")
    for first, count, energy in shells(energies):
        here = min(count, max(0, N_ELECTRONS - filled))
        filled += here
        print(f"  {count:6d}   {energy * ENERGY_UNIT_MEV:12.3f}   {here} of {count}")
    print(f"  gap between level 18 and level 19: {gap(energies) * ENERGY_UNIT_MEV:.4f} meV"
          f"   ({'CLOSED' if gap(energies) > 1e-7 else 'OPEN'} shell)")
    print(f"  energy of the 18 lowest levels: {energies[:N_ELECTRONS].sum() * ENERGY_UNIT_MEV / N_ELECTRONS:.4f} meV per electron")
    print(f"  lowest level of the second band: {second_band * ENERGY_UNIT_MEV:.2f} meV (above all 27 levels of the first)")
    print(f"  momentum grid symmetric under 120-degree rotation: {is_threefold_symmetric(momenta)};"
          f"  under k -> -k: {is_inversion_symmetric(momenta)}")
    if gap(energies) > 1e-7:
        print(f"  total crystal momentum of the 18 filled levels: {np.round(total_momentum(momenta), 6) + 0.0}")
    return momenta, energies


def main():
    rows = []
    for twist in ((0.0, 0.0), (1 / 3, 1 / 3)):
        momenta, energies = describe(*twist)
        for i, (k, energy) in enumerate(zip(momenta, energies)):
            rows.append(dict(twist_1=f"{twist[0]:.6f}", twist_2=f"{twist[1]:.6f}", level=i + 1,
                             kx=f"{k[0]:.6f}", ky=f"{k[1]:.6f}", energy_meV=f"{energy * ENERGY_UNIT_MEV:.6f}",
                             filled=int(i < N_ELECTRONS)))
    print("\nGap (meV) between level 18 and level 19 for twists T1 (down), T2 (across), in sixths of a turn;")
    print("* marks a momentum grid with three-fold symmetry:")
    print("        " + "".join(f"{j}/6      " for j in range(6)))
    for i in range(6):
        cells = []
        for j in range(6):
            momenta, energies, _ = levels(i / 6, j / 6)
            cells.append(f"{gap(energies) * ENERGY_UNIT_MEV:6.3f}{'*' if is_threefold_symmetric(momenta) else ' '}  ")
        print(f"  {i}/6   " + "".join(cells))
    out = Path(__file__).resolve().parent.parent / "results"
    out.mkdir(exist_ok=True)
    with open(out / "one_electron_levels.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {out / 'one_electron_levels.csv'}")


if __name__ == "__main__":
    main()
