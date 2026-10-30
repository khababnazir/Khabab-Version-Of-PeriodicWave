#!/bin/bash
cd /tmp
PY=/home/khabab/moire-nnvmc/.venv/bin/python
for args in "ours logpsi 1024" "ours eloc_folx 1024" "ours eloc_hessian 1024" "ours eloc_folx 4096"; do
    DETERMINANT=slogdet CUDA_VISIBLE_DEVICES=0 $PY pieces18.py $args 2>&1 | grep -E '^\{|Error' | grep -v -i warn | tail -2
    echo "--- slogdet $args: exit ${PIPESTATUS[0]}  $(date +%T)"
done
echo "== all done"
