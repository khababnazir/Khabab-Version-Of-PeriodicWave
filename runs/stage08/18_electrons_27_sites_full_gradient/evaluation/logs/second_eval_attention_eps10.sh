#!/bin/bash
# 2026-11-05: second frozen-weight evaluation of attention_18e_eps10 (full gradient), 10x the
# burn-in (2,000 steps) and seed 1, because the first one's halves differ by 2.7 errors.
# Waits for the GPU 1 training queue (launch_table1.sh full, PID 126529) to finish.
cd ~/moire-nnvmc/learn/stage08_own_code || exit 1
F=../../runs/stage08/18_electrons_27_sites_full_gradient
LOGS=$F/evaluation/logs
mkdir -p $LOGS
echo "=== $(date '+%F %T')  waiting for the GPU 1 queue (PID 126529) to finish"
while kill -0 126529 2>/dev/null; do sleep 60; done
echo "=== $(date '+%F %T')  second frozen-weight evaluation of attention_18e_eps10 on GPU 1: burn-in 2000, seed 1"
stamp=$(date '+%F_%H%M%S')
nvidia-smi -i 1 --format=csv -l 10 \
    --query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,clocks.sm,fan.speed,clocks_event_reasons.active \
    >> $LOGS/gpu1_monitor_$stamp.csv 2>&1 &
gpu_monitor=$!
../../.venv/bin/python system_monitor.py --parent $$ --every 10 --out $LOGS/gpu1_system_$stamp.csv &
system_monitor=$!
CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 ../../.venv/bin/python analysis/evaluate_weights.py \
    $F/training/attention_18e_eps10 --burn-in 2000 --seed 1 --out $F/evaluation/attention_18e_eps10_burnin2000_seed1
echo "=== $(date '+%F %T')  evaluation ended with exit code $?"
kill $gpu_monitor $system_monitor
