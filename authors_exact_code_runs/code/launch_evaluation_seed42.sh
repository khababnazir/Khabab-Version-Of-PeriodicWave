#!/bin/bash
# Frozen-weight evaluation of the seed-42 eps = 3 run (evaluate_authors_weights.py) on GPU 1:
# 20,480,000 configurations (20,000 steps x 1024 walkers).
# Run on 2026-11-23 01:46 with a second line, the same evaluation of the honeycomb-field run on
# GPU 0; that run and its evaluation are now in ../archive/eps3_seed42_honeycomb_field/.
#
#   bash launch_evaluation_seed42.sh
#
# Output: <run folder>/evaluation/<parameters>/.  Logs and monitors (every 10 s) in
# ../logs/evaluation_seed42_gpuN*.
cd "$(dirname "$0")" || exit 1
PY=../../.venv/bin/python
LOGS=../logs
SEED42=../runs/18_electrons_27_sites_eps3_seed42
mkdir -p $LOGS

evaluate () {   # gpu run-folder
    gpu=$1
    run=$2
    stamp=$(date '+%F_%H%M%S')
    nvidia-smi -i $gpu --format=csv -l 10 \
        --query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,clocks.sm,fan.speed,clocks_event_reasons.active \
        >> $LOGS/evaluation_seed42_gpu${gpu}_monitor_$stamp.csv 2>&1 &
    gpu_monitor=$!
    $PY ../../learn/stage08_own_code/system_monitor.py --parent $$ --every 10 --out $LOGS/evaluation_seed42_gpu${gpu}_system_$stamp.csv &
    system_monitor=$!
    echo "=== $(date '+%F %T')  evaluate $run, eps = 3, on GPU $gpu"
    CUDA_VISIBLE_DEVICES=$gpu PYTHONUNBUFFERED=1 $PY evaluate_authors_weights.py $run --inverse-eps 1/3
    status=$?                                  # saved first: $(date) below would overwrite $?
    echo "=== $(date '+%F %T')  evaluation of $run ended with exit code $status"
    kill $gpu_monitor $system_monitor
}

export -f evaluate
export PY LOGS
setsid nohup bash -c "evaluate 1 $SEED42/ordinary" >> $LOGS/evaluation_seed42_gpu1.log 2>&1 < /dev/null &
echo "launched on GPU 1; log in $LOGS/evaluation_seed42_gpu1.log"
