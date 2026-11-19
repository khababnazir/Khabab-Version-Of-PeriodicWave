#!/bin/bash
# Frozen-weight evaluation of the two eps = 3 runs (evaluate_authors_weights.py), one per GPU:
#   GPU 0:  self-attention (CustomPsiformer)     GPU 1:  SlaterNet
# 20,480,000 configurations each (20,000 steps x 1024 walkers).
#
#   bash launch_evaluation_eps3.sh
#
# Output: ../runs/18_electrons_27_sites_eps3_released_settings/evaluation/<parameters>/ (self-attention)
# and .../evaluation/SlaterNet_<parameters>/.  Logs and monitors (every 10 s) in ../logs/evaluation_eps3_gpuN*.
cd "$(dirname "$0")" || exit 1
PY=../../.venv/bin/python
LOGS=../logs
RUN=../runs/18_electrons_27_sites_eps3_released_settings
mkdir -p $LOGS

evaluate () {   # gpu network
    gpu=$1
    network=$2
    stamp=$(date '+%F_%H%M%S')
    nvidia-smi -i $gpu --format=csv -l 10 \
        --query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,clocks.sm,fan.speed,clocks_event_reasons.active \
        >> $LOGS/evaluation_eps3_gpu${gpu}_monitor_$stamp.csv 2>&1 &
    gpu_monitor=$!
    $PY ../../learn/stage08_own_code/system_monitor.py --parent $$ --every 10 --out $LOGS/evaluation_eps3_gpu${gpu}_system_$stamp.csv &
    system_monitor=$!
    echo "=== $(date '+%F %T')  evaluate $network, eps = 3, on GPU $gpu"
    CUDA_VISIBLE_DEVICES=$gpu PYTHONUNBUFFERED=1 $PY evaluate_authors_weights.py $RUN --inverse-eps 1/3 --network $network
    status=$?                                  # saved first: $(date) below would overwrite $?
    echo "=== $(date '+%F %T')  evaluation of $network, eps = 3, ended with exit code $status"
    kill $gpu_monitor $system_monitor
}

export -f evaluate
export PY LOGS RUN
setsid nohup bash -c "evaluate 0 CustomPsiformer" >> $LOGS/evaluation_eps3_gpu0.log 2>&1 < /dev/null &
setsid nohup bash -c "evaluate 1 SlaterNet" >> $LOGS/evaluation_eps3_gpu1.log 2>&1 < /dev/null &
echo "launched on GPU 0 and GPU 1; logs in $LOGS/evaluation_eps3_gpu*.log"
