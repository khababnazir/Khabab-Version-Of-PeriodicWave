#!/bin/bash
# PARADOX EXPERIMENT: one training run with twisted boundary conditions.
#
#   18 electrons on 27 sites, eps = 5, self-attention network (released form), the paper's
#   Table II settings, seed 0, folx + slogdet, --gradient full, and the ONE new thing:
#   --twist 1/3 1/3 (the wave function gets the phase exp(2 pi i / 3) when an electron goes
#   once around the supercell along L1 or L2).  Everything except the twist is the run
#   runs/stage08/18_electrons_27_sites_full_gradient/training/attention_18e_eps5, the baseline.
#
#   bash launch_twisted.sh          start (or continue) the training on GPU 0, then evaluate
#                                   the final weights frozen.  About 44 hours + 1.3 hours.
#   bash launch_twisted.sh watch    start the watchdog (one line every 30 minutes into
#                                   training/logs/watch.log).  Start it once train_stats.csv exists.
#
# Code: train_moire.py and evaluate_weights.py of THIS folder (copies with the twist; see
# ../CHANGES.diff).  system_monitor.py and watch_run.py are used, unchanged, from learn/.
# Everything is written below ../runs/18_electrons_27_sites_twist_one_third_full_gradient/:
#   training/attention_18e_eps5/     settings.json, train_stats.csv, checkpoints, starts.log
#   training/logs/gpu0.log           this queue's log;  gpu0_monitor_*.csv (GPU, every 10 s),
#                                    gpu0_system_*.csv (CPU and memory, every 10 s), watch.log
#   evaluation/attention_18e_eps5/   energies.csv, observables.npz, summary.json
#
# Safe to run again: the training continues from its newest checkpoint, a finished run exits
# at once, and if GPU 0 or the folder is busy the trainer exits with code 75.
#
# Watch:  tail -f ../runs/18_electrons_27_sites_twist_one_third_full_gradient/training/logs/gpu0.log
case "$1" in
    ""|watch) ;;
    *) echo "usage: bash launch_twisted.sh [watch]"; exit 2 ;;
esac
cd "$(dirname "$0")" || exit 1
PY=../../.venv/bin/python
LEARN=../../learn/stage08_own_code
GPU=0
RUN=attention_18e_eps5
RUNS=../runs/18_electrons_27_sites_twist_one_third_full_gradient/training
OPTIONS="--electrons 18 --settings table2 --laplacian folx --determinants slogdet --network attention --eps 5 --gradient full --twist 1/3 1/3"
LOGS=$RUNS/logs
mkdir -p $LOGS

if [ "$1" = "watch" ]; then
    setsid nohup $PY $LEARN/watch_run.py $RUNS/$RUN --every 1800 >> $LOGS/watch.log 2>&1 < /dev/null &
    echo "watchdog started; it writes $LOGS/watch.log"
    exit 0
fi

queue () {
    stamp=$(date '+%F_%H%M%S')
    nvidia-smi -i $GPU --format=csv -l 10 \
        --query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,clocks.sm,fan.speed,clocks_event_reasons.active \
        >> $LOGS/gpu${GPU}_monitor_$stamp.csv 2>&1 &
    gpu_monitor=$!
    $PY $LEARN/system_monitor.py --parent $$ --every 10 --out $LOGS/gpu${GPU}_system_$stamp.csv &
    system_monitor=$!
    echo "=== $(date '+%F %T')  start $RUN on GPU $GPU:  $OPTIONS"
    CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 $PY train_moire.py $OPTIONS --out $RUNS/$RUN
    status=$?                                  # saved first: $(date) below would overwrite $?
    echo "=== $(date '+%F %T')  $RUN ended with exit code $status"
    if [ $status -eq 75 ]; then
        echo "=== GPU $GPU or $RUN is busy; nothing was started"
    elif [ $status -eq 0 ]; then
        echo "=== $(date '+%F %T')  frozen-weight evaluation of $RUN on GPU $GPU"
        CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 $PY evaluate_weights.py $RUNS/$RUN
        status=$?                              # saved first: $(date) below would overwrite $?
        echo "=== $(date '+%F %T')  evaluation of $RUN ended with exit code $status"
    fi
    kill $gpu_monitor $system_monitor
}

export -f queue
export PY LEARN GPU RUN RUNS OPTIONS LOGS
setsid nohup bash -c queue >> $LOGS/gpu$GPU.log 2>&1 < /dev/null &
echo "launched on GPU $GPU; log: $LOGS/gpu$GPU.log"
