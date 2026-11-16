#!/bin/bash
# Frozen-weight evaluation of the four authors'-code runs (evaluate_authors_weights.py), on both GPUs:
#   GPU 0:  released settings eps = 5,  then Table II settings eps = 10
#   GPU 1:  Table II settings eps = 5,  then released settings eps = 10
# (both eps = 5 first).  20,480,000 configurations each.  Measured speeds (2026-11-16, 220-step
# test): released 2.7 steps/s (20,000 steps: about 2 hours), Table II 0.96 steps/s on GPU 1 at its
# 250 W cap (5,000 steps: about 1.5 hours).  About 3.5 to 4 hours in all.
#
#   bash launch_evaluation.sh
#
# Output: <run folder>/evaluation/<the authors' output folder name>/ (energies.csv,
# observables.npz, summary.json).  Each queue logs into ../logs/evaluation_gpuN.log and records,
# every 10 s, its GPU (evaluation_gpuN_monitor_<start>.csv) and CPU/memory
# (evaluation_gpuN_system_<start>.csv).  Safe to run again: a finished evaluation is not
# repeated; a busy GPU or folder makes the evaluator exit with code 75, which stops that queue.
cd "$(dirname "$0")" || exit 1
PY=../../.venv/bin/python
LOGS=../logs
mkdir -p $LOGS

queue () {   # gpu, then "run-folder 1/eps" pairs
    gpu=$1
    shift
    stamp=$(date '+%F_%H%M%S')
    nvidia-smi -i $gpu --format=csv -l 10 \
        --query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,clocks.sm,fan.speed,clocks_event_reasons.active \
        >> $LOGS/evaluation_gpu${gpu}_monitor_$stamp.csv 2>&1 &
    gpu_monitor=$!
    $PY ../../learn/stage08_own_code/system_monitor.py --parent $$ --every 10 --out $LOGS/evaluation_gpu${gpu}_system_$stamp.csv &
    system_monitor=$!
    for job in "$@"; do
        set -- $job
        echo "=== $(date '+%F %T')  evaluate $1, 1/eps = $2, on GPU $gpu"
        CUDA_VISIBLE_DEVICES=$gpu PYTHONUNBUFFERED=1 $PY evaluate_authors_weights.py ../runs/$1 --inverse-eps $2
        status=$?                                  # saved first: $(date) below would overwrite $?
        echo "=== $(date '+%F %T')  evaluation of $1, 1/eps = $2, ended with exit code $status"
        [ $status -eq 75 ] && { echo "=== GPU $gpu or the output folder is busy; this queue stops"; break; }
    done
    kill $gpu_monitor $system_monitor
}

export -f queue
export PY LOGS
setsid nohup bash -c "queue 0 '18_electrons_27_sites_released_settings 0.2' '18_electrons_27_sites_table2_settings 0.1'" \
    >> $LOGS/evaluation_gpu0.log 2>&1 < /dev/null &
setsid nohup bash -c "queue 1 '18_electrons_27_sites_table2_settings 0.2' '18_electrons_27_sites_released_settings 0.1'" \
    >> $LOGS/evaluation_gpu1.log 2>&1 < /dev/null &
echo "launched on GPU 0 and GPU 1; logs in $LOGS/"
