#!/bin/bash
# Stage 8, 2026-11-02: does the energy rise of SlaterNet eps = 10 (18 electrons,
# 27 sites) after about 35,000 iterations come from the gradient term that the
# authors' code leaves out?  Two branches of
#     runs/stage08/18_electrons_27_sites/training/slaternet_18e_eps10
# (only read), both from its checkpoint_033545.npz, made by branch_run.py.
# Only the gradient differs; architecture, batch, sampling, norm constraint
# and every other setting are the original's (train_moire.py checks):
#     GPU 1:  real  (the authors' gradient, as the original, which also ran on GPU 1)
#     GPU 0:  full
# Both train to 76,070 iterations, the original's first checkpoint after the
# rise, so the real branch can be compared with it checkpoint to checkpoint.
#
#   bash learn/stage08_own_code/launch_gradient_test.sh check
#       First, about 3 minutes: 20 iterations of each branch, into
#       gradient_test/restore_check/, compared with the original
#       (branch_run.py compare).  real must repeat the original's rows; full
#       its first row, which is measured before the first update.
#   bash learn/stage08_own_code/launch_gradient_test.sh
#       Then the two runs, into gradient_test/training/ (about 2.5 hours, one
#       per GPU), each followed by its comparison with the original and its
#       frozen-weight evaluation (evaluate_weights.py: 5,000 steps x 4,096
#       walkers, about 20 minutes, into gradient_test/evaluation/).
#   bash learn/stage08_own_code/launch_gradient_test.sh settle
#       Only if an evaluation was still drifting (gradient_test_results.py
#       says): a second frozen-weight evaluation of BOTH runs, with 10x the
#       burn-in (2,000 steps) and another seed (1), into
#       gradient_test/evaluation/<run>_burnin2000_seed1/ (about 25 minutes).
#
# Each queue records its GPU (nvidia-smi) and the machine's CPU and memory
# (system_monitor.py) every 10 s, in the logs/ folder beside the run folders.
# Watch:  tail -f runs/stage08/18_electrons_27_sites/gradient_test/training/logs/gpu1.log
cd "$(dirname "$0")" || exit 1
PY=../../.venv/bin/python
SYSTEM=../../runs/stage08/18_electrons_27_sites
ORIGINAL=$SYSTEM/training/slaternet_18e_eps10
STEP=33545
OPTIONS="--network slaternet --eps 10 --electrons 18 --settings table2 --laplacian folx --determinants slogdet"

branch () {   # gpu, gradient, folder, iterations, evaluate (yes/no)
    gpu=$1 gradient=$2 folder=$3 iterations=$4 evaluate=$5
    logs=$(dirname $folder)/logs
    stamp=$(date '+%F_%H%M%S')
    nvidia-smi -i $gpu --format=csv -l 10 \
        --query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,clocks.sm,fan.speed,clocks_event_reasons.active \
        >> $logs/gpu${gpu}_monitor_$stamp.csv 2>&1 &
    gpu_monitor=$!
    $PY system_monitor.py --parent $$ --every 10 --out $logs/gpu${gpu}_system_$stamp.csv &
    system_monitor=$!
    if [ ! -e $folder/branch.json ]; then
        echo "=== $(date '+%F %T')  make $folder: checkpoint $STEP of $ORIGINAL, gradient $gradient"
        $PY branch_run.py make $ORIGINAL $STEP $folder --gradient $gradient
    fi
    if [ -e $folder/branch.json ]; then
        echo "=== $(date '+%F %T')  train $folder on GPU $gpu to $iterations iterations:  $OPTIONS --gradient $gradient"
        CUDA_VISIBLE_DEVICES=$gpu PYTHONUNBUFFERED=1 $PY train_moire.py $OPTIONS --gradient $gradient \
            --iterations $iterations --out $folder
        status=$?                                  # saved first: $(date) below would overwrite $?
        echo "=== $(date '+%F %T')  training ended with exit code $status"
        echo "=== compared with the original:"
        $PY branch_run.py compare $ORIGINAL $folder
        if [ $status -eq 0 ] && [ "$evaluate" = yes ]; then
            echo "=== $(date '+%F %T')  frozen-weight evaluation of $folder on GPU $gpu"
            CUDA_VISIBLE_DEVICES=$gpu PYTHONUNBUFFERED=1 $PY analysis/evaluate_weights.py $folder
            echo "=== $(date '+%F %T')  evaluation ended with exit code $?"
        fi
    fi
    kill $gpu_monitor $system_monitor
}

settle () {   # gpu, training folder: the second evaluation, 10x burn-in, seed 1
    gpu=$1 folder=$2
    logs=$(dirname $folder)/logs
    stamp=$(date '+%F_%H%M%S')
    nvidia-smi -i $gpu --format=csv -l 10 \
        --query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,clocks.sm,fan.speed,clocks_event_reasons.active \
        >> $logs/gpu${gpu}_monitor_$stamp.csv 2>&1 &
    gpu_monitor=$!
    $PY system_monitor.py --parent $$ --every 10 --out $logs/gpu${gpu}_system_$stamp.csv &
    system_monitor=$!
    out=$SYSTEM/gradient_test/evaluation/$(basename $folder)_burnin2000_seed1
    echo "=== $(date '+%F %T')  second frozen-weight evaluation of $folder on GPU $gpu: burn-in 2000, seed 1"
    CUDA_VISIBLE_DEVICES=$gpu PYTHONUNBUFFERED=1 $PY analysis/evaluate_weights.py $folder \
        --burn-in 2000 --seed 1 --out $out
    echo "=== $(date '+%F %T')  evaluation ended with exit code $?"
    kill $gpu_monitor $system_monitor
}

start () {   # gpu, gradient, folder, iterations, evaluate
    mkdir -p $(dirname $3)/logs
    setsid nohup bash -c "branch $*" >> $(dirname $3)/logs/gpu$1.log 2>&1 < /dev/null &
}

export -f branch settle
export PY SYSTEM ORIGINAL STEP OPTIONS
RUNS=$SYSTEM/gradient_test/training
if [ "$1" = "settle" ]; then
    for pair in "1 $RUNS/slaternet_18e_eps10_from033545_real" "0 $RUNS/slaternet_18e_eps10_from033545_full"; do
        set -- $pair
        setsid nohup bash -c "settle $1 $2" >> $RUNS/logs/gpu$1.log 2>&1 < /dev/null &
    done
    echo "launched: the second evaluations, real on GPU 1, full on GPU 0; logs in $RUNS/logs/"
    exit 0
elif [ "$1" = "check" ]; then
    CHECK=$SYSTEM/gradient_test/restore_check
    start 1 real $CHECK/slaternet_18e_eps10_from033545_real $((STEP + 20)) no
    start 0 full $CHECK/slaternet_18e_eps10_from033545_full $((STEP + 20)) no
elif [ -z "$1" ]; then
    start 1 real $RUNS/slaternet_18e_eps10_from033545_real 76070 yes
    start 0 full $RUNS/slaternet_18e_eps10_from033545_full 76070 yes
else
    echo "usage: bash launch_gradient_test.sh [check | settle]"
    exit 2
fi
echo "launched: real on GPU 1, full on GPU 0; logs in the logs/ folder beside the run folders"
