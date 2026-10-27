#!/bin/bash
# Stage 8 -- evaluate the four trained networks with their weights frozen
# (evaluate_weights.py).  One queue per GPU:
#   GPU 0: psiformer_eps5, then slaternet_eps5
#   GPU 1: psiformer_eps10, then slaternet_eps10
# At 10000 steps a self-attention evaluation takes about 7 minutes and a
# SlaterNet one about 3, so both queues finish in about 12 minutes.
#
# Safe to start twice, by the same kind of locks as the training:
#   - each GPU queue holds runs/stage08/gpu_locks/gpuN.lock -- the SAME lock the
#     training queues of launch_fig5.sh hold -- so an evaluation never shares
#     a GPU with a training run or with another evaluation;
#   - evaluate_weights.py locks its output folder (exit code 75 if taken) and
#     never replaces a finished evaluation: it says so and exits with 0.
# The exit status of this script is 0 only if every evaluation succeeded.
#
# Run:   bash learn/stage08_first_run/analysis/launch_evaluation.sh    (returns when both queues are done)
# Logs:  runs/stage08/evaluation/logs/gpuN.log  (appended to, never truncated)
cd "$(dirname "$0")" || exit 1
PY=../../../.venv/bin/python
RUNS=../../../runs/stage08
LOGS=$RUNS/evaluation/logs
mkdir -p "$LOGS" "$RUNS/gpu_locks"

queue () {   # gpu, then run names.  Returns 0 only if every evaluation succeeded.
    gpu=$1
    shift
    exec 9> "$RUNS/gpu_locks/gpu$gpu.lock"
    if ! flock --nonblock 9; then
        echo "=== $(date -u '+%F %T')  GPU $gpu is in use by another queue (training or evaluation); stopping"
        return 75
    fi
    failed=0
    for run in "$@"; do
        echo "=== $(date -u '+%F %T')  start $run on GPU $gpu"
        CUDA_VISIBLE_DEVICES=$gpu PYTHONUNBUFFERED=1 $PY evaluate_weights.py "$RUNS/training/$run"
        status=$?                                  # saved first: $(date) below would overwrite $?
        echo "=== $(date -u '+%F %T')  $run ended with exit code $status"
        [ $status -eq 0 ] || failed=1
    done
    return $failed
}

queue 0 psiformer_eps5 slaternet_eps5 >> "$LOGS/gpu0.log" 2>&1 &
first=$!
queue 1 psiformer_eps10 slaternet_eps10 >> "$LOGS/gpu1.log" 2>&1 &
second=$!
wait $first;  status0=$?
wait $second; status1=$?
echo "GPU 0 queue exit $status0, GPU 1 queue exit $status1; logs in $LOGS"
[ $status0 -eq 0 ] && [ $status1 -eq 0 ]
