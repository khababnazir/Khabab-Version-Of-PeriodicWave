"""
Stage 4, part 1 -- the local energy  E_loc(R) = (H Psi)(R) / Psi(R).

Paper:   Sec. IV A, Eqs. (24)-(28)
Authors: external/PeriodicWave/periodicwave/pbc/hamiltonian.py

Why the local energy?  The energy of a trial wave function is

    E = <Psi|H|Psi> / <Psi|Psi>
      = average of E_loc(R), over configurations R drawn from |Psi(R)|^2.

So if we can (1) draw configurations from |Psi|^2 and (2) evaluate E_loc at
each one, we get E without doing a 2N-dimensional integral.
Part (2) is this file.  Part (1) is metropolis.py.

What H contains here: kinetic energy + moire potential (the H0 part of the
paper's Eq. 1).  The electron-electron Coulomb repulsion needs Ewald
summation because of the periodic boundary conditions; that is Stage 7.
Until then this is the NON-interacting problem -- still a genuine
many-electron Hamiltonian, and enough to build and test the machinery.

Units, same as the authors' code: lengths in moire lattice constants a_M,
energies in hbar^2 / (m* a_M^2), which is 3.375559 meV for WSe2/WS2.
"""
import sys
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

# Reuse the Stage 3 code (the LogPsi type, the lattice helpers).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "stage03_slater"))
from slaternet import LogPsi  # noqa: E402


# ---------------------------------------------------------------------------
#  Units
# ---------------------------------------------------------------------------

def moire_units(effective_mass=0.35, lattice_constant_nm=8.031, potential_meV=15.0):
    """Convert the paper's material parameters into the code's units.

    Returns (energy_unit_meV, potential_strength).  Same numbers as
    convert_moire_scales() in the authors' run_2deg_triangular_potential.py.
    """
    hbar = 6.582119569509066e-1              # meV * ps
    electron_mass = 5.685630060215049e-3     # meV / (nm/ps)^2

    mass = effective_mass * electron_mass
    energy_unit_meV = hbar**2 / (mass * lattice_constant_nm**2)
    potential_strength = potential_meV / energy_unit_meV
    return energy_unit_meV, potential_strength


# ---------------------------------------------------------------------------
#  Potential energy: the moire potential
# ---------------------------------------------------------------------------

def moire_reciprocal_vectors():
    """The three shortest reciprocal vectors g1, g2, g3 of the moire lattice.

    These are the authors' vectors (make_triangular_potential), pointing at
    90, 330 and 210 degrees.  The paper's formula writes the same star rotated
    by 90 degrees (0, 120, 240): the paper and the code orient the lattice
    differently.  With the code's lattice, a1 = (1, 0), only the code's
    vectors make V periodic.  Mixing the two conventions breaks it silently.
    """
    length = 4 * np.pi / np.sqrt(3)
    g1 = length * np.array([0.0, 1.0])
    g2 = length * np.array([np.sin(2 * np.pi / 3), np.cos(2 * np.pi / 3)])
    g3 = -(g1 + g2)
    return np.stack([g1, g2, g3])


def moire_potential(positions, strength, phi):
    """Eq. (1):  V = sum over electrons i of  -2 V0 sum_j cos(g_j . r_i + phi).

    positions: (N, 2).  Returns one number, summed over all electrons.
    """
    g = jnp.asarray(moire_reciprocal_vectors())     # (3, 2)
    phases = positions @ g.T                         # (N, 3): entry [i, j] = g_j . r_i
    return -2.0 * strength * jnp.sum(jnp.cos(phases + phi))


# ---------------------------------------------------------------------------
#  Kinetic energy
# ---------------------------------------------------------------------------

def kinetic_energy(log_psi_fn, positions):
    """Local kinetic energy  -1/2 (Laplacian Psi) / Psi,  for one configuration.

    log_psi_fn: a function positions (N,2) -> LogPsi(log_abs, phase).

    Write Psi = exp(u + i*phi), with u = log|Psi| and phi = the phase.  Then

        (Laplacian Psi) / Psi = Lap(u) + |grad u|^2 - |grad phi|^2
                                + i * ( Lap(phi) + 2 grad(u) . grad(phi) )

    Every term is a derivative of u or phi, so working with log Psi costs us
    nothing.  The result is complex.  Its real part enters the energy.  Its
    imaginary part averages to zero over |Psi|^2, but is not zero at each
    configuration.  (The authors' fast "folx" path keeps only the real part --
    we keep both, because Stage 6 will need to check what dropping it does.)
    """
    # The Laplacian is over ALL 2N coordinates together, so flatten (N, 2) -> (2N,).
    shape = positions.shape
    x = positions.reshape(-1)

    def u(x):
        return log_psi_fn(x.reshape(shape)).log_abs

    def phi(x):
        return log_psi_fn(x.reshape(shape)).phase

    # First derivatives.
    grad_u = jax.grad(u)(x)
    grad_phi = jax.grad(phi)(x)

    # Second derivatives.  The Laplacian is the trace of the Hessian matrix.
    # Building the whole Hessian is wasteful -- the authors use a "forward
    # Laplacian" (the folx library) that computes the trace directly -- but it
    # is the clearest way to write it, and fast enough for a few electrons.
    laplacian_u = jnp.trace(jax.hessian(u)(x))
    laplacian_phi = jnp.trace(jax.hessian(phi)(x))

    real_part = laplacian_u + jnp.sum(grad_u**2) - jnp.sum(grad_phi**2)
    imag_part = laplacian_phi + 2.0 * jnp.sum(grad_u * grad_phi)

    return -0.5 * (real_part + 1j * imag_part)


# ---------------------------------------------------------------------------
#  Local energy
# ---------------------------------------------------------------------------

def local_energy(log_psi_fn, positions, potential_strength, potential_phi):
    """E_loc = kinetic + potential, for one configuration.  Complex-valued."""
    kinetic = kinetic_energy(log_psi_fn, positions)
    potential = moire_potential(positions, potential_strength, potential_phi)
    return kinetic + potential


# ---------------------------------------------------------------------------
#  A known exact eigenstate, for checking all of the above
# ---------------------------------------------------------------------------

def plane_wave_k_vectors(lattice, n_electrons):
    """The n_electrons shortest wavevectors that fit the supercell.

    A plane wave exp(i k.r) is periodic in the supercell only if
    k = n1 G1 + n2 G2 with integers n1, n2 (G = reciprocal supercell vectors).
    Any set of distinct allowed k gives an exact eigenstate; we pick the
    shortest ones.
    """
    G = 2 * np.pi * np.linalg.inv(lattice)          # rows are G1 and G2
    candidates = []
    for n1 in range(-3, 4):
        for n2 in range(-3, 4):
            candidates.append(n1 * G[0] + n2 * G[1])
    candidates.sort(key=lambda k: np.dot(k, k))
    return np.array(candidates[:n_electrons])


def plane_wave_log_psi(positions, k_vectors):
    """Slater determinant of plane waves, det[ exp(i k_j . r_i) ].

    With no potential this is an exact eigenstate of the kinetic energy, with
    energy  sum_j |k_j|^2 / 2.  So its local energy must be exactly that
    number at EVERY configuration -- the zero-variance property, Eq. (28).
    """
    phases = positions @ jnp.asarray(k_vectors).T    # (N, N): entry [i, j] = k_j . r_i
    matrix = jnp.exp(1j * phases)
    sign, log_abs = jnp.linalg.slogdet(matrix)
    return LogPsi(log_abs=log_abs, phase=jnp.angle(sign))
