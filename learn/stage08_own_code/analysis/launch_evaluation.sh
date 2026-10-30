#!/bin/bash
# Stage 8 (our code) -- evaluate the four trained networks with their weights
# frozen (evaluate_weights.py), one after the other on ONE GPU (default 0):
#   attention_eps5, attention_eps10, slaternet_eps5, slaternet_eps10
# One GPU, not two: on 2026-10-28 GPU 1 dropped off the bus after hours with
# both GPUs at full load, and GPU 0 alone already runs at 87-91 C.  While the
# queue runs, nvidia-smi records both GPUs every 10 s: utilization, memory,
# temperature, power, clock, fan, and why the clock is lowered; and
# system_monitor.py records the CPU and memory use of the machine and of the
# evaluation.
#
# Safe to start twice: every evaluate_weights.py takes the lock of its GPU
# (runs/gpu_locks/gpuN.lock, which training holds too) and of its output
# folder, and exits with code 75 if either is taken.  The queue then stops,
# rather than starting its next evaluation on a busy GPU.  A finished
# evaluation is not repeated (exit code 0).  The exit status of this script
# is 0 only if every evaluation succeeded.
#
# Run (from ~/moire-nnvmc):  bash learn/stage08_own_code/analysis/launch_evaluation.sh [GPU]
# Logs:  runs/stage08/evaluation/logs/gpuN.log (appended to, never truncated)
#        runs/stage08/evaluation/logs/gpu_monitor_<date>_<time>.csv, one file per start
#        runs/stage08/evaluation/logs/system_monitor_<date>_<time>.csv, beside it
cd "$(dirname "$0")" || exit 1
GPU=${1:-0}
PY=../../../.venv/bin/python
RUNS=../../../runs/stage08
LOGS=$RUNS/evaluation/logs
ALREADY_RUNNING=75
mkdir -p "$LOGS"

stamp=$(date '+%F_%H%M%S')
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,clocks.sm,fan.speed,clocks_event_reasons.active \
    --format=csv -l 10 >> "$LOGS/gpu_monitor_$stamp.csv" 2>&1 &
monitor=$!
$PY ../system_monitor.py --parent $$ --every 10 --out "$LOGS/system_monitor_$stamp.csv" &
system_monitor=$!

failed=0
for run in attention_eps5 attention_eps10 slaternet_eps5 slaternet_eps10; do
    echo "=== $(date '+%F %T')  start $run on GPU $GPU"
    CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 $PY evaluate_weights.py "$RUNS/training/$run"
    status=$?                                  # saved first: $(date) below would overwrite $?
    echo "=== $(date '+%F %T')  $run ended with exit code $status"
    if [ $status -eq $ALREADY_RUNNING ]; then
        echo "=== GPU $GPU or $run is busy; the queue stops"
        failed=1
        break
    fi
    [ $status -eq 0 ] || failed=1
done >> "$LOGS/gpu$GPU.log" 2>&1

kill $monitor $system_monitor
exit $failed
