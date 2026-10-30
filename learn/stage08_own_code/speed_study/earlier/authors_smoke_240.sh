#!/bin/bash
PY=/home/khabab/moire-nnvmc/.venv/bin/python
run () {   # name, GPUs, batch, iterations
    name=$1; gpus=$2; batch=$3; its=$4
    out=/tmp/authors_smoke/$name; rm -rf $out $out.log $out.progress; mkdir -p $out
    echo "=== $(date +%T) start $name: GPUs $gpus, batch $batch, $its iterations"
    XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=$gpus PYTHONUNBUFFERED=1 \
        $PY /tmp/authors_smoke.py $batch $its $out CustomPsiformer > $out.log 2>&1 &
    pid=$!
    while kill -0 $pid 2>/dev/null; do
        rows=0
        [ -f "$out/train_stats.csv" ] && rows=$(( $(wc -l < "$out/train_stats.csv") - 1 ))
        echo "$(date +%s.%N) $rows $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | tr '\n' ' ')" >> $out.progress
        sleep 2
    done
    wait $pid; status=$?
    echo "=== $(date +%T) $name exit $status"
    grep -E -i 'out of memory|RESOURCE_EXHAUSTED|Error' $out.log | grep -v -i warn | head -3
}
run two_gpus_4096_240 0,1 4096 240
run one_gpu_1024_240 0 1024 240
run one_gpu_4096_240 0 4096 240
echo "== all done"
