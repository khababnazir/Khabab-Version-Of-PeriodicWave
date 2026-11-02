#!/bin/bash
# Stage 8 (our code) -- evaluate trained networks with their weights frozen
# (evaluate_weights.py), one after the other on ONE GPU (default 0).
#
# Without a run list: the four 9-site runs of Fig. 5,
#   training/attention_eps5 training/attention_eps10 training/slaternet_eps5 training/slaternet_eps10
# With a run list: those folders under runs/stage08, e.g. table1/attention_eps5.
# RUN@N evaluates checkpoint_N.npz of RUN instead of its final checkpoint.
# For the Table I runs (2026-11-02) start one queue per GPU:
#   bash learn/stage08_own_code/analysis/launch_evaluation.sh 0 table1/attention_eps5 table1/slaternet_eps5
#   bash learn/stage08_own_code/analysis/launch_evaluation.sh 1 table1/attention_eps10 table1/slaternet_eps10 table1/slaternet_eps10@33545
#
# While the queue runs, nvidia-smi records both GPUs every 10 s: utilization,
# memory, temperature, power, clock, fan, and why the clock is lowered; and
# system_monitor.py records the CPU and memory use of the machine and of the
# evaluation.  (The 9-site evaluations ran on one GPU only, after GPU 1 dropped
# off the bus on 2026-10-28; the Table I training then ran 2 days on both.)
#
# Safe to start twice: every evaluate_weights.py takes the lock of its GPU
# (runs/gpu_locks/gpuN.lock, which training holds too) and of its output
# folder, and exits with code 75 if either is taken.  The queue then stops,
# rather than starting its next evaluation on a busy GPU.  A finished
# evaluation is not repeated (exit code 0).  The exit status of this script
# is 0 only if every evaluation succeeded.
#
# Run (from ~/moire-nnvmc):  bash learn/stage08_own_code/analysis/launch_evaluation.sh [GPU] [RUN[@N] ...]
# Logs, in runs/stage08/evaluation/logs (9 sites) or runs/stage08/table1/evaluation/logs (Table I):
#        gpuN.log (appended to, never truncated)
#        gpuN_monitor_<date>_<time>.csv (nvidia-smi), one file per start
#        gpuN_system_<date>_<time>.csv (CPU and memory), beside it
cd "$(dirname "$0")" || exit 1
GPU=${1:-0}
shift
RUN_LIST=("$@")
[ ${#RUN_LIST[@]} -eq 0 ] && RUN_LIST=(training/attention_eps5 training/attention_eps10 training/slaternet_eps5 training/slaternet_eps10)
PY=../../../.venv/bin/python
RUNS=../../../runs/stage08
case "${RUN_LIST[0]}" in
    table1/*) LOGS=$RUNS/table1/evaluation/logs ;;
    *)        LOGS=$RUNS/evaluation/logs ;;
esac
ALREADY_RUNNING=75
mkdir -p "$LOGS"

stamp=$(date '+%F_%H%M%S')
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,clocks.sm,fan.speed,clocks_event_reasons.active \
    --format=csv -l 10 >> "$LOGS/gpu${GPU}_monitor_$stamp.csv" 2>&1 &
monitor=$!
$PY ../system_monitor.py --parent $$ --every 10 --out "$LOGS/gpu${GPU}_system_$stamp.csv" &
system_monitor=$!

failed=0
for item in "${RUN_LIST[@]}"; do
    run=${item%@*}                             # table1/slaternet_eps10@33545 -> table1/slaternet_eps10
    checkpoint=()
    [ "$item" != "$run" ] && checkpoint=(--checkpoint "${item#*@}")
    echo "=== $(date '+%F %T')  start $item on GPU $GPU"
    CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 $PY evaluate_weights.py "$RUNS/$run" "${checkpoint[@]}"
    status=$?                                  # saved first: $(date) below would overwrite $?
    echo "=== $(date '+%F %T')  $item ended with exit code $status"
    if [ $status -eq $ALREADY_RUNNING ]; then
        echo "=== GPU $GPU or $run is busy; the queue stops"
        failed=1
        break
    fi
    [ $status -eq 0 ] || failed=1
done >> "$LOGS/gpu$GPU.log" 2>&1

kill $monitor $system_monitor
exit $failed
