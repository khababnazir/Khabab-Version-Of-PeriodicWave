#!/bin/bash
# Stage 8: the four runs behind the paper's Fig. 5 (6 electrons, 9 sites),
# with the authors' code and configuration, 200,000 iterations each.
#
#   GPU 0:  self-attention eps = 5,   then SlaterNet eps = 5
#   GPU 1:  self-attention eps = 10,  then SlaterNet eps = 10
#
# One process per GPU (see the 2026-10-01 crash note in progress/2026-10-08.txt).
# Each run takes about 8 hours (7 iterations per second on an RTX 3090).
#
# Restarting.  Nothing restarts a run automatically.  After a crash or a
# reboot, run this script again BY HAND: every run resumes from the last
# checkpoint in its folder (written every 30 minutes), and a finished run
# exits almost immediately.  It is safe to run while jobs are still alive:
# run_moire.py refuses to start in a folder another process is training into.
#
# The two runs on one GPU are independent experiments, so if the first one
# fails, the log says FAILED with its exit code and the second one still runs.
#
# Usage:   bash learn/stage08_first_run/launch_fig5.sh        (from ~/moire-nnvmc)
# Watch:   tail -f runs/stage08/logs/gpu0.log
#          tail -1 runs/stage08/psiformer_eps5/train_stats.csv
set -e
cd "$(dirname "$0")"
PY=../../.venv/bin/python
RUNS=../../runs/stage08
mkdir -p $RUNS/logs

run () {   # gpu network eps folder
    echo "=== $(date '+%F %T')  start $4 on GPU $1"
    if CUDA_VISIBLE_DEVICES=$1 PYTHONUNBUFFERED=1 $PY run_moire.py --network $2 --eps $3 --out $RUNS/$4; then
        echo "=== $(date '+%F %T')  finished $4"
    else
        status=$?                     # save it: the $(date) below would reset $?
        echo "=== $(date '+%F %T')  FAILED $4 (exit code $status)"
    fi
}

export -f run
export PY RUNS
setsid nohup bash -c "run 0 CustomPsiformer 5 psiformer_eps5;   run 0 SlaterNet 5 slaternet_eps5"   >> $RUNS/logs/gpu0.log 2>&1 &
setsid nohup bash -c "run 1 CustomPsiformer 10 psiformer_eps10; run 1 SlaterNet 10 slaternet_eps10" >> $RUNS/logs/gpu1.log 2>&1 &
echo "launched; logs in runs/stage08/logs/"
