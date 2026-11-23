#!/bin/bash
# Does the seed override do what it says?  On GPU 0, three runs of 3 iterations each, in /tmp:
#   SCREEN_SEED=42  must give their seed-42 start: row 0 of train_stats.csv as the seed-42 run of
#                   authors_exact_code_runs (pmove 0.16832682 exactly), and a config.json that differs
#                   from that run's only in optim.iterations;
#   SCREEN_SEED=1   must give another start (another pmove) with the same config.json;
#   a MUTANT copy without the override line, SCREEN_SEED=1: must look like seed 42 (and so be caught).
#   bash check_seed_override.sh        (about 6 minutes; GPU 0 must be free)
cd "$(dirname "$0")" || exit 1
HERE=$PWD
REPO=$HOME/moire-nnvmc
PY=$REPO/.venv/bin/python
ARGS="18_0 27 0.35 0.3333333333333333 8.031 15 45 CustomPsiformer"
T=/tmp/screen_check
rm -rf $T && mkdir -p $T/s42 $T/s1 $T/mutant
$PY -c "import sys; t = open(sys.argv[1]).read(); line = 'jax.random.PRNGKey = _screen_prng_key\n'; \
assert t.count(line) == 1; open(sys.argv[2], 'w').write(t.replace(line, ''))" \
    run_2deg_triangular_potential_screen.py $T/mutant_script.py || { echo "mutant not made"; exit 1; }
[ $(diff run_2deg_triangular_potential_screen.py $T/mutant_script.py | grep -c '^[<>]') -eq 1 ] || { echo "mutant not made"; exit 1; }
run () {   # folder seed script
    (cd $1 && SCREEN_SEED=$2 SCREEN_ITERATIONS=3 CUDA_VISIBLE_DEVICES=0 $PY $3 $ARGS > run.log 2>&1)
    echo "run in $1 (SCREEN_SEED=$2): exit $?"
}
run $T/s42 42 $HERE/run_2deg_triangular_potential_screen.py
run $T/s1 1 $HERE/run_2deg_triangular_potential_screen.py
run $T/mutant 1 $T/mutant_script.py
$PY - <<'EOF'
import json
from pathlib import Path
OUT = "results/2deg-CoulombMoire/CustomPsiformer/el18_0_N27_V4.44370908_45.0_U17.70578894"
SEED42 = Path.home() / "moire-nnvmc/authors_exact_code_runs/runs/18_electrons_27_sites_eps3_seed42/ordinary" / OUT
T = Path("/tmp/screen_check")
def row0(folder):
    lines = (folder / "train_stats.csv").read_text().splitlines()
    head, first = lines[0].split(","), lines[1].split(",")
    return {k: float(v) for k, v in zip(head, first)}
def flat(d, p=""):
    out = {}
    for k, v in d.items():
        out.update(flat(v, p + k + ".") if isinstance(v, dict) else {p + k: v})
    return out
ref = row0(SEED42)
r42, r1, rm = (row0(T / n / OUT) for n in ("s42", "s1", "mutant"))
c_ref = flat(json.loads((SEED42 / "config.json").read_text()))
c42, c1 = (flat(json.loads((T / n / OUT / "config.json").read_text())) for n in ("s42", "s1"))
print(f"pmove at iteration 0: seed-42 run {ref['pmove']}, SCREEN_SEED=42 {r42['pmove']}, SCREEN_SEED=1 {r1['pmove']}, mutant {rm['pmove']}")
print(f"energy at iteration 0: seed-42 run {ref['energy']}, SCREEN_SEED=42 {r42['energy']}, SCREEN_SEED=1 {r1['energy']}")
config_diff = sorted(k for k in set(c_ref) | set(c42) if c_ref.get(k) != c42.get(k))
checks = {"SCREEN_SEED=42 starts as the seed-42 run (pmove)": r42["pmove"] == ref["pmove"],
          "SCREEN_SEED=42 starts as the seed-42 run (energy within 1e-3)": abs(r42["energy"] - ref["energy"]) < 1e-3,
          "SCREEN_SEED=1 starts elsewhere": r1["pmove"] != ref["pmove"] and abs(r1["energy"] - ref["energy"]) > 1e-3,
          "config.json differs from the seed-42 run's only in optim.iterations": config_diff == ["optim.iterations"],
          "SCREEN_SEED=1 and 42 have the same config.json": c1 == c42,
          "the mutant (no override) is caught: it starts as seed 42": rm["pmove"] == ref["pmove"]}
for name, ok in checks.items():
    print(("PASS " if ok else "FAIL ") + name)
print("config differences:", config_diff)
raise SystemExit(0 if all(checks.values()) else 1)
EOF
