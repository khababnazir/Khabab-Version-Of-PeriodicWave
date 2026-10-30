#!/bin/bash
cd /tmp
PY=/home/khabab/moire-nnvmc/.venv/bin/python
nvidia-smi -i 0 --format=csv -l 5 --query-gpu=timestamp,utilization.gpu,memory.used,temperature.gpu,power.draw,clocks_event_reasons.active > /tmp/bench18_gpu.csv 2>&1 &
monitor=$!
T2='{"n_layers": 3, "n_heads": 6, "perceptrons_per_layer": 1}'
for args in "attention 6 9 1024" "slaternet 18 27 1024" "attention 18 27 1024" "attention 18 27 1024 T2" "attention 18 27 4096 T2" "slaternet 18 27 4096"; do
    set -- $args
    if [ "$5" = "T2" ]; then CUDA_VISIBLE_DEVICES=0 $PY bench18.py $1 $2 $3 $4 "$T2"; else CUDA_VISIBLE_DEVICES=0 $PY bench18.py $1 $2 $3 $4; fi 2>&1 | grep -E '^\{|Error|error' | grep -v -i warn | tail -2
    echo "--- exit ${PIPESTATUS[0]}  $(date +%T)"
done
kill $monitor
echo "== all done"
