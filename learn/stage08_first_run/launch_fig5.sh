#!/bin/bash
# Stage 8: the four runs behind the paper's Fig. 5 (6 electrons, 9 sites),
# with the authors' code and configuration, 200,000 iterations each.
#
#   GPU 0 queue:  self-attention eps = 5,   then SlaterNet eps = 5
#   GPU 1 queue:  self-attention eps = 10,  then SlaterNet eps = 10
#
# One process per GPU (see the 2026-10-01 crash note in progress/2026-10-08.txt).
# Each run takes about 8 hours (7 iterations per second on an RTX 3090).
#
# Restarting.  Nothing restarts a run automatically.  After a crash or a
# reboot, run this script again BY HAND: every run resumes from the last
# checkpoint in its folder (written every 30 minutes), and a finished run
# exits at once.
#
# Running it while jobs are alive does no harm, by two locks:
#   - each GPU queue holds runs/stage08/gpu_locks/gpuN.lock while it runs, so a
#     second queue for the same GPU stops immediately;
#   - run_moire.py exits with code 75 if its folder is already being trained
#     The queue then stops too, rather than starting its next run on a GPU
#     that is busy.
# A run that fails for any other reason is logged as FAILED with its exit
# code, and the queue goes on to its next run, which is independent.
#
# Usage:   bash learn/stage08_first_run/launch_fig5.sh        (from ~/moire-nnvmc)
# Watch:   tail -f runs/stage08/training/logs/gpu0.log
#          tail -1 runs/stage08/training/psiformer_eps5/train_stats.csv
ALREADY_RUNNING=75

run () {   # gpu network eps folder;  returns run_moire.py's exit code
    echo "=== $(date '+%F %T')  start $4 on GPU $1"
    CUDA_VISIBLE_DEVICES=$1 PYTHONUNBUFFERED=1 $PY run_moire.py --network $2 --eps $3 --out $RUNS/training/$4
    status=$?                         # save it: the $(date) below would reset $?
    if [ $status -eq 0 ]; then
        echo "=== $(date '+%F %T')  finished $4"
    elif [ $status -eq $ALREADY_RUNNING ]; then
        echo "=== $(date '+%F %T')  $4 is already being trained by another process"
    else
        echo "=== $(date '+%F %T')  FAILED $4 (exit code $status)"
    fi
    return $status
}

queue () {   # gpu, then groups of three: network eps folder
    gpu=$1; shift
    exec 9> "$RUNS/gpu_locks/gpu$gpu.lock"
    if ! flock --nonblock 9; then
        echo "=== $(date '+%F %T')  a queue for GPU $gpu is already running; this one stops"
        return 0
    fi
    while [ $# -gt 0 ]; do
        result=0
        run $gpu $1 $2 $3 || result=$?
        if [ $result -eq $ALREADY_RUNNING ]; then
            echo "=== $(date '+%F %T')  GPU $gpu is busy; this queue stops"
            return 0
        fi
        shift 3
    done
}

# `source launch_fig5.sh --functions-only` loads the two functions and nothing
# else, so they can be tested with stand-in programs.
if [ "${1:-}" = "--functions-only" ]; then return 0 2>/dev/null || exit 0; fi

set -e
cd "$(dirname "$0")"
PY=../../.venv/bin/python
RUNS=../../runs/stage08
mkdir -p $RUNS/training/logs $RUNS/gpu_locks

export -f run queue
export PY RUNS ALREADY_RUNNING
setsid nohup bash -c "queue 0 CustomPsiformer 5 psiformer_eps5   SlaterNet 5 slaternet_eps5"   >> $RUNS/training/logs/gpu0.log 2>&1 &
setsid nohup bash -c "queue 1 CustomPsiformer 10 psiformer_eps10 SlaterNet 10 slaternet_eps10" >> $RUNS/training/logs/gpu1.log 2>&1 &
echo "launched; logs in runs/stage08/training/logs/"
