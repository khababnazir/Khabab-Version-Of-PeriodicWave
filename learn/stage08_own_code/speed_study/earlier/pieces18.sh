#!/bin/bash
cd /tmp
PY=/home/khabab/moire-nnvmc/.venv/bin/python
nvidia-smi -i 0 --format=csv -l 5 --query-gpu=timestamp,utilization.gpu,memory.used,temperature.gpu,power.draw,clocks_event_reasons.active > /tmp/pieces18_gpu.csv 2>&1 &
monitor=$!
for args in "authors logpsi 1024" "authors eloc_folx 1024" "ours logpsi 1024" "ours eloc_hessian 1024" "ours eloc_folx 1024" "ours move 1024" "ours kfac 1024"; do
    CUDA_VISIBLE_DEVICES=0 $PY pieces18.py $args 2>&1 | grep -E '^\{|Error' | grep -v -i warn | tail -2
    echo "--- $args: exit ${PIPESTATUS[0]}  $(date +%T)"
done
kill $monitor
echo "== all done"
