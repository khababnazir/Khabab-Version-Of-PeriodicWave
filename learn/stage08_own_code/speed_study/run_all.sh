#!/bin/bash
# Stage 8 speed study: every measurement on GPU 0 only, one process at a time.
# Output: speed_results.jsonl (one JSON line per run), logs/<run>.log (full output), gpu_usage.csv (both GPUs, every 2 s),
#         system_usage.csv (CPU and memory of the machine and of the runs, every 2 s; added after the 30 October run).
cd "$(dirname "$0")"
PY=/home/khabab/moire-nnvmc/.venv/bin/python
mkdir -p logs
: > speed_results.jsonl
nvidia-smi --format=csv -l 2 --query-gpu=timestamp,index,utilization.gpu,memory.used,temperature.gpu,power.draw,clocks_event_reasons.active > gpu_usage.csv 2>&1 &
monitor=$!
$PY ../system_monitor.py --parent $$ --every 2 --out system_usage.csv &
system_monitor=$!
run() {   # run NAME ENV... -- COMMAND...
    name=$1; shift
    echo "=== $(date +%T) start $name" >> progress.txt
    env CUDA_VISIBLE_DEVICES=0 "$@" > logs/$name.log 2>&1
    status=$?
    line=$(grep '^{' logs/$name.log | tail -1)
    if [ -z "$line" ]; then line="{\"run\": \"$name\", \"error\": \"no result line, exit $status: $(grep -E 'Error|error' logs/$name.log | grep -v -i warn | tail -1 | tr -d '\"' | cut -c1-300)\"}"; fi
    echo "{\"run\": \"$name\", \"exit\": $status, \"result\": $line}" >> speed_results.jsonl
    echo "=== $(date +%T) done  $name exit $status" >> progress.txt
}
for n in 6 18; do for route in folx_det folx_slogdet hessian_det hessian_slogdet; do for b in 64 1024; do
    run det_alone_${n}e_${route}_b$b timeout 900 $PY det_alone.py $n $route $b
done; done; done
for piece in logpsi eloc_hessian eloc_folx; do run pieces_6e_ours_det_$piece N_EL=6 timeout 900 $PY pieces.py ours $piece 1024; done
for piece in eloc_hessian eloc_folx move kfac; do run pieces_6e_ours_slogdet_$piece N_EL=6 DETERMINANT=slogdet timeout 900 $PY pieces.py ours $piece 1024; done
for piece in logpsi eloc_folx; do run pieces_6e_authors_$piece N_EL=6 timeout 900 $PY pieces.py authors $piece 1024; done
run full_6e_attention_folx_det timeout 900 $PY full_iteration.py attention 6 9 1024
run full_6e_attention_folx_slogdet DETERMINANT=slogdet timeout 900 $PY full_iteration.py attention 6 9 1024
for net in attention slaternet; do for b in 1024 4096; do
    run full_18e_${net}_folx_slogdet_b$b DETERMINANT=slogdet timeout 900 $PY full_iteration.py $net 18 27 $b
done; done
# CPU-only checks (added when the study moved into the repo): singular determinants and what folx sees
for p in float64 float32; do JAX_PLATFORMS=cpu $PY singular_example.py $p 2>/dev/null | grep '^{'; done |
    $PY -c "import json, sys; json.dump([json.loads(l) for l in sys.stdin], open('singular_example.json', 'w'), indent=1)"
JAX_PLATFORMS=cpu $PY jaxpr_probe.py > logs/jaxpr_probe.log 2>&1
kill $monitor $system_monitor
echo "== all done $(date +%T)" >> progress.txt
