#!/bin/bash
# eps = 3, the authors' code, their released settings, the self-attention network, FIXED SEED 42:
# their script with one line added (run_2deg_triangular_potential_seed42.py; changes.diff).
#
# Started on 2026-11-19 with the earlier version of this script (`bash how_this_was_started.sh 1
# ordinary`).  That version had a second mode, the honeycomb-field test on GPU 0; on 2026-11-23 the
# test was moved to ../../archive/eps3_seed42_honeycomb_field/ (the earlier script is there, in
# run_folder_before_the_move/).  This version runs exactly what the "ordinary" mode ran.
#
# Start (G = 0 or 1), from this folder:
#   setsid nohup flock -n ~/moire-nnvmc/runs/gpu_locks/gpuG.lock bash how_this_was_started.sh G >> logs/gpuG.log 2>&1 < /dev/null &
# Safe to start again: their code continues from its newest checkpoint.
# Around the training, only: one GPU visible, two passive monitors every 10 s (logs/), and a check
# that the GPU can start a program.
cd "$(dirname "$0")" || exit 1
USAGE="usage: bash how_this_was_started.sh GPU"
[ $# -eq 1 ] || { echo "$USAGE"; exit 2; }
case "$1" in 0|1) GPU=$1 ;; *) echo "$USAGE"; exit 2 ;; esac
HERE=$PWD
REPO=$HOME/moire-nnvmc
PY=${PY:-$REPO/.venv/bin/python}
ARGS="18_0 27 0.35 0.3333333333333333 8.031 15 45 CustomPsiformer"     # 1/eps = 1/3, as the first eps = 3 run
mkdir -p logs ordinary

if ! CUDA_VISIBLE_DEVICES=$GPU $REPO/.venv/bin/python -c \
        "import ctypes, sys; sys.exit(0 if ctypes.CDLL('libcuda.so.1').cuInit(0) == 0 else 1)"; then
    echo "=== $(date '+%F %T')  GPU $GPU cannot start a program (the driver refuses; reboot needed?). Nothing started."
    exit 1
fi

stamp=$(date +%F_%H%M%S)
nvidia-smi -i $GPU --format=csv -l 10 \
    --query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,clocks.sm,fan.speed,clocks_event_reasons.active \
    >> logs/gpu${GPU}_monitor_$stamp.csv 2>&1 &
gpu_monitor=$!
$PY $REPO/learn/stage08_own_code/system_monitor.py --parent $$ --every 10 --out logs/gpu${GPU}_system_$stamp.csv &
system_monitor=$!

echo "=== $(date '+%F %T')  start run_2deg_triangular_potential_seed42.py in ordinary on GPU $GPU"
(cd ordinary && CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 $PY "$HERE/run_2deg_triangular_potential_seed42.py" $ARGS)
status=$?    # saved first: $(date) below would overwrite $?
echo "=== $(date '+%F %T')  run_2deg_triangular_potential_seed42.py in ordinary ended with exit code $status"
kill $gpu_monitor $system_monitor
