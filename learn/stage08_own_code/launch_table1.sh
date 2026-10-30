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
#       Then the four runs.  Measured in a 40-iteration test on GPU 0 alone:
#       attention network 0.82 s per iteration, so 150,000 iterations take
#       about 34 hours (longer if the GPUs slow down when hot).  SlaterNet
#       about 0.18 s per iteration (5-iteration timing): 7-8 hours.
#
# Safe to run again: train_moire.py continues every run from its newest
# checkpoint, a finished run exits at once, and a run whose folder or GPU is
# busy exits with code 75, which stops its queue.  A run that fails for any
# other reason is logged with its exit code, and the queue goes on.
#
# Each queue records, every 10 s, one pair of files per start, in runs/stage08/table1/logs/:
#   gpuN_monitor_<date>_<time>.csv   its GPU: utilization, memory, temperature, power, clock, fan, slowdowns
#   gpuN_system_<date>_<time>.csv    CPU and memory, of the machine and of the run (system_monitor.py)
#
# Watch:  tail -f runs/stage08/table1/logs/gpu0.log
cd "$(dirname "$0")" || exit 1
PY=../../.venv/bin/python
RUNS=../../runs/stage08/table1
LOGS=$RUNS/logs
TABLE1="--electrons 18 --settings table2 --laplacian folx --determinants slogdet"
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
    done
    kill $gpu_monitor $system_monitor
}

start () {   # gpu, then the runs of its queue
    setsid nohup bash -c "queue $*" >> $LOGS/gpu$1.log 2>&1 < /dev/null &
}

export -f queue
export PY RUNS LOGS TABLE1
if [ "$1" = "test" ]; then
    start 0 "'lr_test/eta10 --network attention --eps 5 --learning-rate 10 --iterations 2000'"
    start 1 "'lr_test/eta0.1 --network attention --eps 5 --learning-rate 0.1 --iterations 2000'"
elif [ -z "$1" ]; then
    start 0 "'attention_eps5 --network attention --eps 5'" "'slaternet_eps5 --network slaternet --eps 5'"
    start 1 "'attention_eps10 --network attention --eps 10'" "'slaternet_eps10 --network slaternet --eps 10'"
else
    echo "usage: bash launch_table1.sh [test]"
    exit 2
fi
echo "launched, one run on GPU 0 and one on GPU 1; logs in runs/stage08/table1/logs/"
