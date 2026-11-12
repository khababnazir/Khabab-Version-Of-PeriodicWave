#!/bin/bash
# Stage 8, 2026-11-02: does the energy rise of SlaterNet eps = 10 (18 electrons,
# 27 sites) after about 35,000 iterations come from the gradient term that the
# authors' code leaves out?  Two branches of
#     runs/stage08/18_electrons_27_sites_real_gradient/training/slaternet_18e_eps10
# (only read), both from its checkpoint_033545.npz, made by branch_run.py.
# Only the gradient differs; architecture, batch, sampling, norm constraint
# and every other setting are the original's (train_moire.py checks):
#     GPU 1:  real  (the authors' gradient, as the original, which also ran on GPU 1)
#     GPU 0:  full
# Both train to 76,070 iterations, the original's first checkpoint after the
# rise, so the real branch can be compared with it checkpoint to checkpoint.
# Every evaluation folder says which checkpoint it evaluated: <run>_at_<iterations>.
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
#       walkers, about 20 minutes, into gradient_test/evaluation/<run>_at_076070/).
#   bash learn/stage08_own_code/launch_gradient_test.sh settle
#       Only if an evaluation was still drifting (gradient_test_results.py
#       says): a second frozen-weight evaluation of BOTH runs at 76,070, with
#       10x the burn-in (2,000 steps) and another seed (1), into
#       gradient_test/evaluation/<run>_at_076070_burnin2000_seed1/ (about 25 minutes).
#   bash learn/stage08_own_code/launch_gradient_test.sh extend          (2026-11-03)
#       The full branch on from 76,070 to 150,000, the original's end, on GPU 0
#       alone (about 4.5 hours), then its comparison with the original and its
#       frozen-weight evaluation into gradient_test/evaluation/<run>_at_150000/.
#       The real branch is not continued: it repeats the original exactly.
#   bash learn/stage08_own_code/launch_gradient_test.sh settle 150000
#       The same second evaluation for the full branch at 150,000, if needed.
#
# Each queue records its GPU (nvidia-smi) and the machine's CPU and memory
# (system_monitor.py) every 10 s, in the logs/ folder beside the run folders.
# Watch:  tail -f runs/stage08/18_electrons_27_sites_real_gradient/gradient_test/training/logs/gpu0.log
cd "$(dirname "$0")" || exit 1
PY=../../.venv/bin/python
SYSTEM=../../runs/stage08/18_electrons_27_sites_real_gradient
ORIGINAL=$SYSTEM/training/slaternet_18e_eps10
EVALUATION=$SYSTEM/gradient_test/evaluation
STEP=33545
OPTIONS="--network slaternet --eps 10 --electrons 18 --settings table2 --laplacian folx --determinants slogdet"

monitors () {   # gpu, logs folder: GPU and system records every 10 s, until killed
    stamp=$(date '+%F_%H%M%S')
    nvidia-smi -i $1 --format=csv -l 10 \
        --query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,clocks.sm,fan.speed,clocks_event_reasons.active \
        >> $2/gpu${1}_monitor_$stamp.csv 2>&1 &
    gpu_monitor=$!
    $PY system_monitor.py --parent $$ --every 10 --out $2/gpu${1}_system_$stamp.csv &
    system_monitor=$!
}

branch () {   # gpu, gradient, folder, iterations, evaluate (yes/no)
    gpu=$1 gradient=$2 folder=$3 iterations=$4 evaluate=$5
    monitors $gpu $(dirname $folder)/logs
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
            out=$EVALUATION/$(basename $folder)_at_$(printf %06d $iterations)
            echo "=== $(date '+%F %T')  frozen-weight evaluation of $folder at $iterations on GPU $gpu, into $out"
            CUDA_VISIBLE_DEVICES=$gpu PYTHONUNBUFFERED=1 $PY analysis/evaluate_weights.py $folder \
                --checkpoint $iterations --out $out
            status=$?    # saved first: $(date) below would overwrite $?
            echo "=== $(date '+%F %T')  evaluation ended with exit code $status"
        fi
    fi
    kill $gpu_monitor $system_monitor
}

settle () {   # gpu, training folder, checkpoint: the second evaluation, 10x burn-in, seed 1
    gpu=$1 folder=$2 checkpoint=$3
    monitors $gpu $(dirname $folder)/logs
    out=$EVALUATION/$(basename $folder)_at_$(printf %06d $checkpoint)_burnin2000_seed1
    echo "=== $(date '+%F %T')  second frozen-weight evaluation of $folder at $checkpoint on GPU $gpu: burn-in 2000, seed 1"
    CUDA_VISIBLE_DEVICES=$gpu PYTHONUNBUFFERED=1 $PY analysis/evaluate_weights.py $folder \
        --checkpoint $checkpoint --burn-in 2000 --seed 1 --out $out
    status=$?    # saved first: $(date) below would overwrite $?
    echo "=== $(date '+%F %T')  evaluation ended with exit code $status"
    kill $gpu_monitor $system_monitor
}

start () {   # logs folder, function, gpu, the function's other arguments
    logs=$1
    shift
    mkdir -p $logs
    setsid nohup bash -c "$*" >> $logs/gpu$2.log 2>&1 < /dev/null &
}

export -f monitors branch settle
export PY SYSTEM ORIGINAL EVALUATION STEP OPTIONS
RUNS=$SYSTEM/gradient_test/training
REAL=$RUNS/slaternet_18e_eps10_from033545_real
FULL=$RUNS/slaternet_18e_eps10_from033545_full
if [ "$1" = "check" ]; then
    CHECK=$SYSTEM/gradient_test/restore_check
    start $CHECK/logs branch 1 real $CHECK/slaternet_18e_eps10_from033545_real $((STEP + 20)) no
    start $CHECK/logs branch 0 full $CHECK/slaternet_18e_eps10_from033545_full $((STEP + 20)) no
    echo "launched: real on GPU 1, full on GPU 0"
elif [ -z "$1" ]; then
    start $RUNS/logs branch 1 real $REAL 76070 yes
    start $RUNS/logs branch 0 full $FULL 76070 yes
    echo "launched: real on GPU 1, full on GPU 0"
elif [ "$1" = "settle" ] && [ -z "$2" ]; then
    start $RUNS/logs settle 1 $REAL 76070
    start $RUNS/logs settle 0 $FULL 76070
    echo "launched: the second evaluations at 76,070, real on GPU 1, full on GPU 0"
elif [ "$1" = "extend" ]; then
    start $RUNS/logs branch 0 full $FULL 150000 yes
    echo "launched: the full branch to 150,000 on GPU 0"
elif [ "$1" = "settle" ] && [ "$2" = "150000" ]; then
    start $RUNS/logs settle 0 $FULL 150000
    echo "launched: the second evaluation of the full branch at 150,000 on GPU 0"
else
    echo "usage: bash launch_gradient_test.sh [check | settle | extend | settle 150000]"
    exit 2
fi
echo "logs in the logs/ folder beside the run folders"
