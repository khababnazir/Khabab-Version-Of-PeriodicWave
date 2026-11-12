#!/bin/bash
# The authors' released code (external/PeriodicWave at commit 6525d2f3eec53f085b55d485e1bff45d61ef1782,
# no file of it changed) with the paper's Table II values: started with a COPY of their script,
# run_2deg_triangular_potential_table2.py in this folder, in which seven numbers are changed
# (changes.diff: walkers 4096, 150,000 iterations, learning rate 10 / (1 + t/100000), 3 layers,
# 6 heads, 1 perceptron per layer).  Everything else is theirs, their random seed included.
# 18 electrons on 27 sites, the self-attention network (CustomPsiformer), eps = 10; or, with a
# second argument 0.2 (added 2026-11-12), eps = 5.
#
#   cd runs/stage08/18_electrons_27_sites_authors_code_table2
#   setsid nohup flock -n ~/moire-nnvmc/runs/gpu_locks/gpuG.lock bash how_this_was_started.sh G [0.1 | 0.2] >> logs/gpuG.log 2>&1 < /dev/null &
# with G = 0 or 1 (flock holds the lock our own tools use on that GPU).
#
# Around the training, only: one GPU made visible, the output kept in logs/, two passive monitors
# every 10 s (nvidia-smi, system_monitor.py), and a check before the run that the GPU can start a
# program at all (their code would otherwise run on the CPU without a word).  Their script writes
# into results/2deg-CoulombMoire/CustomPsiformer/<parameters>/ below this folder and continues
# from its last checkpoint if this is started again.
cd "$(dirname "$0")" || exit 1
USAGE="usage: bash how_this_was_started.sh GPU [INVERSE_EPS]    (GPU: 0 or 1;  1/eps: 0.1 (default) or 0.2)"
case "$1" in 0|1) GPU=$1 ;; *) echo "$USAGE"; exit 2 ;; esac
case "${2:-0.1}" in 0.1|0.2) INVERSE_EPS=${2:-0.1} ;; *) echo "$USAGE"; exit 2 ;; esac
[ $# -le 2 ] || { echo "$USAGE"; exit 2; }
REPO=$HOME/moire-nnvmc
PY=${PY:-$REPO/.venv/bin/python}
mkdir -p logs

gpu_can_start_a_program () {    # asks the driver only (cuInit); creates nothing on the GPU
    CUDA_VISIBLE_DEVICES=$GPU $REPO/.venv/bin/python -c \
        "import ctypes, sys; sys.exit(0 if ctypes.CDLL('libcuda.so.1').cuInit(0) == 0 else 1)"
}

if ! gpu_can_start_a_program; then
    echo "=== $(date '+%F %T')  GPU $GPU cannot start a program (the driver refuses; reboot needed?). Nothing started."
    exit 1
fi
stamp=$(date +%F_%H%M%S)
nvidia-smi -i $GPU --format=csv -l 10 \
    --query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,clocks.sm,fan.speed,clocks_event_reasons.active \
    >> logs/gpu${GPU}_monitor_$stamp.csv 2>&1 &
gpu_monitor=$!
$REPO/.venv/bin/python $REPO/learn/stage08_own_code/system_monitor.py --parent $$ --every 10 --out logs/gpu${GPU}_system_$stamp.csv &
system_monitor=$!
SCRIPT=$PWD/run_2deg_triangular_potential_table2.py
echo "=== $(date '+%F %T')  start 1/eps = $INVERSE_EPS (Table II values) on GPU $GPU"
CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 $PY $SCRIPT 18_0 27 0.35 $INVERSE_EPS 8.031 15 45 CustomPsiformer
status=$?    # saved first: $(date) below would overwrite $?
echo "=== $(date '+%F %T')  1/eps = $INVERSE_EPS (Table II values) ended with exit code $status"
kill $gpu_monitor $system_monitor
