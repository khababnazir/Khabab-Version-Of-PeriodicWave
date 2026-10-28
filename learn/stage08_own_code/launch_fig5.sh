#!/bin/bash
# Stage 8: the four runs of the paper's Fig. 5 with our own code (train_moire.py).
# One queue per GPU, each run after the other:
#   GPU 0:  SlaterNet eps = 10,  then the attention network eps = 10
#   GPU 1:  SlaterNet eps = 5,   then the attention network eps = 5
# Measured on an RTX 3090: SlaterNet about 41 iterations per second, the
# attention network about 12, so 200,000 iterations take about 1.5 and 5 hours.
#
# Safe to run again: train_moire.py continues every run from its newest
# checkpoint, a finished run exits at once, and a run whose folder or GPU is
# busy exits with code 75, which stops its queue.  A run that fails for any
# other reason is logged with its exit code, and the queue goes on.
#
# Run (from ~/moire-nnvmc):   bash learn/stage08_own_code/launch_fig5.sh
# Watch:                      tail -f runs/stage08/training/logs/gpu0.log
cd "$(dirname "$0")" || exit 1
PY=../../.venv/bin/python
RUNS=../../runs/stage08/training
mkdir -p $RUNS/logs

queue () {   # gpu, then pairs: network eps
    gpu=$1
    shift
    while [ $# -gt 0 ]; do
        name="${1}_eps${2}"
        echo "=== $(date '+%F %T')  start $name on GPU $gpu"
        CUDA_VISIBLE_DEVICES=$gpu PYTHONUNBUFFERED=1 $PY train_moire.py --network $1 --eps $2 --out $RUNS/$name
        status=$?                                  # saved first: $(date) below would overwrite $?
        echo "=== $(date '+%F %T')  $name ended with exit code $status"
        if [ $status -eq 75 ]; then
            echo "=== GPU $gpu or $name is busy; this queue stops"
            return
        fi
        shift 2
    done
}

export -f queue
export PY RUNS
setsid nohup bash -c "queue 0 slaternet 10 attention 10" >> $RUNS/logs/gpu0.log 2>&1 < /dev/null &
setsid nohup bash -c "queue 1 slaternet 5 attention 5"   >> $RUNS/logs/gpu1.log 2>&1 < /dev/null &
echo "launched; logs in runs/stage08/training/logs/"
