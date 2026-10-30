#!/bin/bash
cd /tmp
PY=/home/khabab/moire-nnvmc/.venv/bin/python
nvidia-smi -i 0 --format=csv -l 5 --query-gpu=timestamp,utilization.gpu,memory.used,temperature.gpu,power.draw,clocks_event_reasons.active > /tmp/bench18_folx_sparse_gpu.csv 2>&1 &
monitor=$!
for args in "attention 18 27 1024" "attention 18 27 4096" "slaternet 18 27 1024" "slaternet 18 27 4096"; do
    set -- $args
    FOLX_SPARSITY=0.6 CUDA_VISIBLE_DEVICES=0 $PY bench18_folx.py $1 $2 $3 $4 2>&1 | grep -E '^\{|Error|error' | grep -v -i warn | tail -2
    echo "--- exit ${PIPESTATUS[0]}  $(date +%T)"
done
kill $monitor
echo "== all done"
