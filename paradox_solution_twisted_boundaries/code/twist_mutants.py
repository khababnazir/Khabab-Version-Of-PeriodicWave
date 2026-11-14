"""PARADOX EXPERIMENT -- do the checks notice mistakes?
Run, from this folder:   ../../.venv/bin/python twist_mutants.py      (CPU, about 15 minutes)

Each "mutant" plants one mistake in a copy of code/ and runs test_twist.py on it.
A mutant is NOTICED if the tests fail; SURVIVED would mean the tests have a hole.  Copies live in <repo>/.mutant_twist/<name>/ (two levels below
the repository, like code/) and are deleted at the end."""
import shutil, subprocess, sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

CODE = Path(__file__).resolve().parent
REPO = CODE.parents[1]
WORK = REPO / ".mutant_twist"
PY = REPO / ".venv" / "bin" / "python"
MUTANTS = [
    ("sign_of_the_phase", "train_moire.py", "phase=psi.phase + jnp.sum(", "phase=psi.phase - jnp.sum("),
    ("twist_on_one_electron_only", "train_moire.py", "jnp.sum(R @ jnp.asarray(k_twist", "jnp.sum(R[:1] @ jnp.asarray(k_twist"),
    ("reciprocal_vectors_swapped", "train_moire.py", "return twist_1 * G[0] + twist_2 * G[1]", "return twist_1 * G[1] + twist_2 * G[0]"),
    ("twist_never_applied", "train_moire.py", "twisted = settings.twist_1 != 0 or settings.twist_2 != 0", "twisted = False"),
    ("twist_only_if_both_nonzero", "train_moire.py", "twisted = settings.twist_1 != 0 or settings.twist_2 != 0", "twisted = settings.twist_1 != 0 and settings.twist_2 != 0"),
    ("command_line_drops_the_twist", "train_moire.py", "network_form=args.network_form,\n                                   twist_1=twist_1, twist_2=twist_2)", "network_form=args.network_form)"),
    ("command_line_swaps_the_twists", "train_moire.py", "for text in args.twist)", "for text in reversed(args.twist))"),
    ("old_folders_not_periodic", "train_moire.py", "{**BEFORE_18_ELECTRONS, **BEFORE_TWIST, **saved[\"settings\"]}", "{**BEFORE_18_ELECTRONS, **saved[\"settings\"]}"),
    ("evaluation_imports_learns_trainer", "evaluate_weights.py", "sys.path.insert(0, str(HERE))             # train_moire.py of THIS folder", "sys.path.insert(0, str(HERE.parents[1] / \"learn\" / \"stage08_own_code\"))  #"),
    ("launcher_without_twist", "launch_twisted.sh", " --gradient full --twist 1/3 1/3\"", " --gradient full\""),
    ("launcher_real_gradient", "launch_twisted.sh", " --gradient full --twist 1/3 1/3\"", " --gradient real --twist 1/3 1/3\""),
    ("shells_wrong_potential_phase", "twist_shells.py", "PHI = np.pi / 4", "PHI = -np.pi / 4"),
]

def run(mutant):
    name, file, old, new = mutant
    folder = WORK / name
    shutil.copytree(CODE, folder, ignore=shutil.ignore_patterns("__pycache__", "twist_mutants.py"))
    text = (folder / file).read_text()
    assert text.count(old) == 1, (name, text.count(old))
    (folder / file).write_text(text.replace(old, new))
    result = subprocess.run(["nice", str(PY), "test_twist.py"], cwd=folder, capture_output=True, text=True)
    failed = sorted({line.split("(")[0].split()[-1] for line in result.stderr.splitlines() + result.stdout.splitlines()
                     if line.startswith(("FAIL:", "ERROR:"))})
    return name, result.returncode, failed

if WORK.exists():
    shutil.rmtree(WORK)
WORK.mkdir()
try:
    with ThreadPoolExecutor(4) as pool:
        for name, code, failed in pool.map(run, MUTANTS):
            print(f"{'NOTICED ' if code != 0 else 'SURVIVED'}  {name}:  {', '.join(failed) or 'no test failed'}", flush=True)
finally:
    shutil.rmtree(WORK)
print("done", flush=True)
