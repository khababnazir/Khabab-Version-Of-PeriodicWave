#!/bin/bash
# Stage 8: the 18-electron runs of the paper's Table I (27 sites), with our own
# code (train_moire.py), the paper's Table II settings, and the local energy
# from folx + slogdet.  One run per GPU at a time, each with its own 4096 walkers:
#   GPU 0:  attention network eps = 5,   then SlaterNet eps = 5
#   GPU 1:  attention network eps = 10,  then SlaterNet eps = 10
#
#   bash learn/stage08_own_code/launch_table1.sh test
#       First, about 30 minutes: 2,000 iterations of the attention network at
#       eps = 5 on both GPUs from the same start (seed 0): learning rate
#       10 / (1 + t/100000) on GPU 0 (Table II), 0.1 / (1 + t/100000) on GPU 1.
#       A stability check and a timing measurement, not a contest.
#   bash learn/stage08_own_code/launch_table1.sh
#       The four runs with the authors' gradient (--gradient real: the
#       imaginary part of the local energy left out, as in their code), into
#       runs/stage08/18_electrons_27_sites_real_gradient/training/.  Run
#       2026-10-30 to 11-01 (attention about 42 hours, SlaterNet about 9).
#   bash learn/stage08_own_code/launch_table1.sh full          (2026-11-03)
#       The same four runs, everything identical (seed 0, Table II, folx +
#       slogdet) except --gradient full (the imaginary part kept), into
#       runs/stage08/18_electrons_27_sites_full_gradient/training/.  Each run
#       is evaluated with frozen weights right after it (evaluate_weights.py,
#       into ../evaluation/<run>/).  About 2.2 days per GPU.
#
# Safe to run again: train_moire.py continues every run from its newest
# checkpoint, a finished run exits at once, and a run whose folder or GPU is
# busy exits with code 75, which stops its queue.  A run that fails for any
# other reason is logged with its exit code, and the queue goes on.
#
# Each queue records, every 10 s, one pair of files per start, in <system>/training/logs/:
#   gpuN_monitor_<date>_<time>.csv   its GPU: utilization, memory, temperature, power, clock, fan, slowdowns
#   gpuN_system_<date>_<time>.csv    CPU and memory, of the machine and of the run (system_monitor.py)
#
# Watch:  tail -f runs/stage08/18_electrons_27_sites_full_gradient/training/logs/gpu0.log
case "$1" in
    ""|test|full) ;;
    *) echo "usage: bash launch_table1.sh [test | full]"; exit 2 ;;
esac
cd "$(dirname "$0")" || exit 1
PY=../../.venv/bin/python
TABLE1="--electrons 18 --settings table2 --laplacian folx --determinants slogdet"
EVALUATE=no
if [ "$1" = "full" ]; then
    RUNS=../../runs/stage08/18_electrons_27_sites_full_gradient/training
    TABLE1="$TABLE1 --gradient full"
    EVALUATE=yes
else
    RUNS=../../runs/stage08/18_electrons_27_sites_real_gradient/training
fi
LOGS=$RUNS/logs
mkdir -p $LOGS

queue () {   # gpu, then one "folder train_moire.py-options" string per run
    gpu=$1
    shift
    stamp=$(date '+%F_%H%M%S')
    nvidia-smi -i $gpu --format=csv -l 10 \
        --query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,clocks.sm,fan.speed,clocks_event_reasons.active \
        >> $LOGS/gpu${gpu}_monitor_$stamp.csv 2>&1 &
    gpu_monitor=$!
    $PY system_monitor.py --parent $$ --every 10 --out $LOGS/gpu${gpu}_system_$stamp.csv &
    system_monitor=$!
    for run in "$@"; do
        set -- $run                                # the folder, then the options
        folder=$1
        shift
        echo "=== $(date '+%F %T')  start $folder on GPU $gpu:  $TABLE1 $*"
        CUDA_VISIBLE_DEVICES=$gpu PYTHONUNBUFFERED=1 $PY train_moire.py $TABLE1 "$@" --out $RUNS/$folder
        status=$?                                  # saved first: $(date) below would overwrite $?
        echo "=== $(date '+%F %T')  $folder ended with exit code $status"
        if [ $status -eq 75 ]; then
            echo "=== GPU $gpu or $folder is busy; this queue stops"
            break
        fi
        if [ $status -eq 0 ] && [ "$EVALUATE" = yes ]; then
            echo "=== $(date '+%F %T')  frozen-weight evaluation of $folder on GPU $gpu"
            CUDA_VISIBLE_DEVICES=$gpu PYTHONUNBUFFERED=1 $PY analysis/evaluate_weights.py $RUNS/$folder
            echo "=== $(date '+%F %T')  evaluation of $folder ended with exit code $?"
        fi
    done
    kill $gpu_monitor $system_monitor
}

start () {   # gpu, then the runs of its queue
    setsid nohup bash -c "queue $*" >> $LOGS/gpu$1.log 2>&1 < /dev/null &
}

export -f queue
export PY RUNS LOGS TABLE1 EVALUATE
if [ "$1" = "test" ]; then
    start 0 "'../lr_test/attention_18e_eps5_eta10 --network attention --eps 5 --learning-rate 10 --iterations 2000'"
    start 1 "'../lr_test/attention_18e_eps5_eta0.1 --network attention --eps 5 --learning-rate 0.1 --iterations 2000'"
elif [ -z "$1" ] || [ "$1" = "full" ]; then
    start 0 "'attention_18e_eps5 --network attention --eps 5'" "'slaternet_18e_eps5 --network slaternet --eps 5'"
    start 1 "'attention_18e_eps10 --network attention --eps 10'" "'slaternet_18e_eps10 --network slaternet --eps 10'"
else
    echo "usage: bash launch_table1.sh [test | full]"
    exit 2
fi
echo "launched, one run on GPU 0 and one on GPU 1; logs in $LOGS/"
