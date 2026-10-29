"""
Stage 8 (our code) -- does a trained network respect the moire lattice's translation symmetry?

Run (on the CPU, a few minutes for all four runs; it never touches a GPU,
so it is safe while a training run is going):
    ../../../.venv/bin/python check_translation_symmetry.py

The archived check (learn/archive/stage08_authors_code/analysis), for our runs
and our networks.  The Hamiltonian does not change when every electron is
moved by the same moire lattice vector a (the moire potential repeats with
period a; kinetic and Coulomb energy depend only on differences).  A wave
function with that symmetry satisfies |Psi(R + a)| = |Psi(R)| for every
configuration R, so log|Psi| would not change at all.  We move the 256 first
walkers of the final checkpoint by a1, a2 and a1 + a2 and print how much
log|Psi| changes.

Control: moving ONE electron by a whole supercell vector must change nothing
(the network is built periodic); only float32 rounding remains.

Why it matters.  If the evaluation finds uneven site occupations (figures 3
and 6), that could be the wave function, or walkers that never mixed between
translated copies of a pattern.  If log|Psi| changes under a lattice
translation, it is the wave function.
"""
import os
import sys
from pathlib import Path

os.environ["JAX_PLATFORMS"] = "cpu"      # before JAX is imported: always off the GPUs

import numpy as np

HERE = Path(__file__).resolve().parent
RUNS = HERE.parents[2] / "runs" / "stage08" / "training"
sys.path.insert(0, str(HERE))
from evaluate_weights import final_checkpoint     # noqa: E402  (puts train_moire.py on the path)
import train_moire                                 # noqa: E402
WALKERS = 256


def main():
    import jax
    import slaternet

    jax.config.update("jax_default_matmul_precision", "float32")     # as the training
    a1, a2 = np.array([1.0, 0.0]), np.array([-0.5, np.sqrt(3) / 2])        # moire lattice, a_M = 1
    supercell_vector = np.asarray(slaternet.supercell_vectors(9))[:, 0]
    for name in ("attention_eps10", "slaternet_eps10", "attention_eps5", "slaternet_eps5"):
        settings, final = final_checkpoint(RUNS / name)
        problem = train_moire.make_problem(settings["network"], settings["eps"])
        like = problem.init(jax.random.PRNGKey(0), problem.arch)
        _, params, walkers, _ = train_moire.load_parameters(final, like)
        x = walkers[:WALKERS]
        log_abs_psi = jax.jit(jax.vmap(lambda R: problem.log_psi(params, R).log_abs))
        reference = log_abs_psi(x)

        def change_after_moving(shift, electrons=slice(None)):
            return np.asarray(log_abs_psi(x.at[:, electrons, :].add(shift)) - reference)

        print(f"{name}")
        print(f"   control: electron 0 moved by a supercell vector   "
              f"max |change| {np.abs(change_after_moving(supercell_vector, 0)).max():.1e}")
        for label, shift in (("a1", a1), ("a2", a2), ("a1 + a2", a1 + a2)):
            change = change_after_moving(shift)
            print(f"   all electrons moved by {label:8s} change of log|Psi|: mean {change.mean():+.3f}, "
                  f"spread {change.std():.3f}, max |change| {np.abs(change).max():.3f}")


if __name__ == "__main__":
    main()
