#!/bin/bash
# Frozen-weight evaluation of the two seed-42 eps = 3 runs (evaluate_authors_weights.py), one per GPU,
# both with their own Hamiltonian only (no field):
#   GPU 0:  honeycomb_field/phase2_field_off   (field for iterations 0-19,999, then off to 199,999)
#   GPU 1:  ordinary                           (no field, 200,000 iterations)
# 20,480,000 configurations each (20,000 steps x 1024 walkers).
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
setsid nohup bash -c "evaluate 0 $SEED42/honeycomb_field/phase2_field_off" >> $LOGS/evaluation_seed42_gpu0.log 2>&1 < /dev/null &
setsid nohup bash -c "evaluate 1 $SEED42/ordinary" >> $LOGS/evaluation_seed42_gpu1.log 2>&1 < /dev/null &
echo "launched on GPU 0 and GPU 1; logs in $LOGS/evaluation_seed42_gpu*.log"
