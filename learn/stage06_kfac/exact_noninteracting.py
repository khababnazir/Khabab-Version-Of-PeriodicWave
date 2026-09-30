"""
Stage 6, part 0 -- an exact answer to aim at.

Until Stage 7 adds the Coulomb repulsion, the electrons do not interact.
Then the exact N-electron ground state is a single Slater determinant of the
N lowest one-electron states, and its energy is the sum of the N lowest
eigenvalues of the one-electron Hamiltonian

    h = -1/2 Laplacian + V(r)       (supercell, periodic boundary conditions)

We get those eigenvalues by writing h in a basis of plane waves exp(i q.r)
that fit the supercell (q = n1 G1 + n2 G2) and diagonalising the matrix:

    kinetic:    <q| -1/2 Laplacian |q>  = |q|^2 / 2
    potential:  V(r) = -2 V0 sum_j cos(g_j.r + phi)
                     = -V0 sum_j [ e^{+i phi} e^{+i g_j.r} + e^{-i phi} e^{-i g_j.r} ]
                so  <q|V|q'> = -V0 e^{+i phi}  if q - q' = +g_j
                               -V0 e^{-i phi}  if q - q' = -g_j,   else 0.

Each moire vector g_j is itself an integer combination of G1 and G2, so the
potential only ever connects basis states to other basis states.

This is the benchmark for training: SlaterNet trained on this Hamiltonian
must reach this energy -- and, being variational, can never go below it.
Stage 9's exact diagonalisation starts from the same plane-wave idea.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "stage04_mcmc"))
from local_energy import moire_reciprocal_vectors  # noqa: E402


def plane_wave_basis(lattice, cutoff):
    """All q = n1 G1 + n2 G2 with |q| <= cutoff.  Returns (integer pairs (n1, n2), vectors q)."""
    G = 2 * np.pi * np.linalg.inv(lattice)                 # rows are G1, G2
    shortest = min(np.linalg.norm(G[0]), np.linalg.norm(G[1]))
    n_max = int(2 * cutoff / shortest) + 1                 # generous: we filter by |q| below

    pairs = []
    for n1 in range(-n_max, n_max + 1):
        for n2 in range(-n_max, n_max + 1):
            q = n1 * G[0] + n2 * G[1]
            if np.dot(q, q) <= cutoff**2:
                pairs.append((n1, n2))
    pairs = np.array(pairs)
    return pairs, pairs @ G


def single_particle_hamiltonian(lattice, strength, phi, cutoff):
    """The matrix <q| h |q'> in the plane-wave basis.  Hermitian, complex."""
    pairs, q = plane_wave_basis(lattice, cutoff)
    position_of = {tuple(p): i for i, p in enumerate(pairs)}

    H = np.diag(0.5 * np.sum(q**2, axis=1)).astype(complex)

    # Each moire vector g_j as integers (m1, m2) with g_j = m1 G1 + m2 G2.
    # Since G_i . L_j = 2 pi delta_ij, the coefficient is m_i = g_j . L_i / (2 pi).
    moire_as_integers = []
    for g in moire_reciprocal_vectors():
        m = g @ lattice / (2 * np.pi)
        assert np.allclose(m, np.round(m)), "moire vector does not fit the supercell"
        moire_as_integers.append(np.round(m).astype(int))

    for i, p in enumerate(pairs):
        for m in moire_as_integers:
            j = position_of.get(tuple(p - m))              # q_j = q_i - g   =>   q_i - q_j = +g
            if j is not None:
                H[i, j] += -strength * np.exp(+1j * phi)
                H[j, i] += -strength * np.exp(-1j * phi)   # the Hermitian partner, q_j - q_i = -g

    assert np.allclose(H, H.conj().T)
    return H


def exact_ground_state_energy(lattice, n_electrons, strength, phi, cutoff=30.0):
    """Sum of the n_electrons lowest one-electron eigenvalues.  Returns (energy, all eigenvalues)."""
    H = single_particle_hamiltonian(lattice, strength, phi, cutoff)
    eigenvalues = np.linalg.eigvalsh(H)
    return float(np.sum(eigenvalues[:n_electrons])), eigenvalues


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "stage03_slater"))
    from slaternet import supercell_vectors
    from local_energy import moire_units

    energy_unit_meV, strength = moire_units()
    phi = np.pi / 4
    lattice = supercell_vectors(9)
    n_electrons = 6

    print("cutoff   basis size   exact E0 (code units)")
    for cutoff in (10.0, 15.0, 20.0, 25.0, 30.0):
        n_basis = len(plane_wave_basis(lattice, cutoff)[0])
        e0, levels = exact_ground_state_energy(lattice, n_electrons, strength, phi, cutoff)
        print(f"{cutoff:6.1f}   {n_basis:10d}   {e0:+.10f}")

    print(f"\nE0 = {e0:+.8f} code units = {e0 * energy_unit_meV:+.4f} meV "
          f"= {e0 * energy_unit_meV / n_electrons:+.4f} meV per electron")
    print("lowest 9 levels:", np.round(levels[:9], 5))
    print(f"gap between level {n_electrons} and {n_electrons + 1}: {levels[n_electrons] - levels[n_electrons - 1]:.3e}"
          "   (zero would mean an open shell: degenerate ground state, same energy)")
