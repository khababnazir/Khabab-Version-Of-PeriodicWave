#!/bin/bash
# Stage 8 -- evaluate the four trained networks with their weights frozen
# (evaluate_weights.py).  One queue per GPU:
#   GPU 0: psiformer_eps5, then slaternet_eps5
#   GPU 1: psiformer_eps10, then slaternet_eps10
# At 10000 steps a self-attention evaluation takes about 7 minutes and a
# SlaterNet one about 3, so both queues finish in about 12 minutes.
#
# Safe to start twice: every evaluate_weights.py takes the lock of its GPU
# (runs/gpu_locks/gpuN.lock, which the training queues of
# launch_fig5.sh hold too) and of its output folder, and exits with code 75
# if either is taken.  A queue that sees 75 stops, rather than starting its
# next evaluation on a busy GPU.  A finished evaluation is not repeated
# (exit code 0).  The exit status of this script is 0 only if every
# evaluation succeeded.
#
# Run:   bash learn/archive/stage08_authors_code/analysis/launch_evaluation.sh    (returns when both queues are done)
# Logs:  runs/archive/stage08_authors_code/evaluation/logs/gpuN.log  (appended to, never truncated)
cd "$(dirname "$0")" || exit 1
PY=../../../../.venv/bin/python
RUNS=../../../../runs/archive/stage08_authors_code
LOGS=$RUNS/evaluation/logs
ALREADY_RUNNING=75
mkdir -p "$LOGS"

queue () {   # gpu, then run names.  Returns 0 only if every evaluation succeeded.
    gpu=$1
    shift
    failed=0
    for run in "$@"; do
        echo "=== $(date -u '+%F %T')  start $run on GPU $gpu"
        CUDA_VISIBLE_DEVICES=$gpu PYTHONUNBUFFERED=1 $PY evaluate_weights.py "$RUNS/training/$run"
        status=$?                                  # saved first: $(date) below would overwrite $?
        echo "=== $(date -u '+%F %T')  $run ended with exit code $status"
        if [ $status -eq $ALREADY_RUNNING ]; then
            echo "=== $(date -u '+%F %T')  GPU $gpu or $run is busy; this queue stops"
            return $ALREADY_RUNNING
        fi
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
