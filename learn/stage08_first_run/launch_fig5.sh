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
# This script is also the restart command: after a crash or reboot, run it
# again.  Every run resumes from the last checkpoint in its folder (written
# every 30 minutes); a finished run exits almost immediately.
#
# Usage:   bash learn/stage08_first_run/launch_fig5.sh        (from ~/moire-nnvmc)
# Watch:   tail -f runs/stage08/logs/gpu0.log
set -e
cd "$(dirname "$0")"
PY=../../.venv/bin/python
RUNS=../../runs/stage08
mkdir -p $RUNS/logs

run () {   # gpu network eps folder
    echo "=== $(date '+%F %T')  start $4 on GPU $1"
    CUDA_VISIBLE_DEVICES=$1 PYTHONUNBUFFERED=1 $PY run_moire.py --network $2 --eps $3 --out $RUNS/$4
    echo "=== $(date '+%F %T')  finished $4"
}

export -f run
export PY RUNS
setsid nohup bash -c "run 0 CustomPsiformer 5 psiformer_eps5;   run 0 SlaterNet 5 slaternet_eps5"   >> $RUNS/logs/gpu0.log 2>&1 &
setsid nohup bash -c "run 1 CustomPsiformer 10 psiformer_eps10; run 1 SlaterNet 10 slaternet_eps10" >> $RUNS/logs/gpu1.log 2>&1 &
echo "launched; logs in runs/stage08/logs/"
