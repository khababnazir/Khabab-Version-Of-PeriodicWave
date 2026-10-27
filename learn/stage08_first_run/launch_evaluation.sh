#!/bin/bash
# Stage 8 -- evaluate the four trained networks with their weights frozen
# (evaluate_weights.py).  One queue per GPU, as for the training:
#   GPU 0: psiformer_eps5, then slaternet_eps5
#   GPU 1: psiformer_eps10, then slaternet_eps10
# At 10000 steps an evaluation takes about 7 minutes, so both queues finish
# in about 15.  Results: runs/stage08/evaluation/<run>/ ; logs in .../evaluation/logs/.
#
# Run:  nohup bash launch_evaluation.sh > /dev/null 2>&1 &
cd "$(dirname "$0")" || exit 1
RUNS=../../runs/stage08
mkdir -p "$RUNS/evaluation/logs"

queue() {
    gpu=$1
    shift
    for run in "$@"; do
        echo "=== $(date -u '+%F %H:%M:%S')  start $run on GPU $gpu"
        CUDA_VISIBLE_DEVICES=$gpu PYTHONUNBUFFERED=1 ../../.venv/bin/python evaluate_weights.py "$RUNS/$run"
        status=$?                                  # saved first: $(date) below would overwrite $?
        echo "=== $(date -u '+%F %H:%M:%S')  $run ended with exit code $status"
    done
}

queue 0 psiformer_eps5 slaternet_eps5 > "$RUNS/evaluation/logs/gpu0.log" 2>&1 &
queue 1 psiformer_eps10 slaternet_eps10 > "$RUNS/evaluation/logs/gpu1.log" 2>&1 &
wait
