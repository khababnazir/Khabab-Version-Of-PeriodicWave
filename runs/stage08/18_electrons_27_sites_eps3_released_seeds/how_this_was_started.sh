#!/bin/bash
# OUR code, eps = 3, 18 electrons on 27 sites, the self-attention network with the authors' RELEASED
# settings (the settings of the authors'-code runs in authors_exact_code_runs/): two random starts,
# one per GPU.  Asked for by Khabab on 2026-11-23, after the authors' code found the honeycomb at
# eps = 3 from its seed 42 (authors_exact_code_runs/runs/18_electrons_27_sites_eps3_seed42/).
#
#   bash how_this_was_started.sh GPU SEED          (used: 0 42 and 1 43)
#
# One run: learn/stage08_own_code/train_moire.py, unchanged, with
#   --network attention --eps 3 --electrons 18 --settings released --laplacian folx
#   --determinants slogdet --gradient real --seed SEED
# into training/attention_18e_eps3_seedSEED/; then, only if it ends with exit code 0, its
# frozen-weight evaluation (learn/stage08_own_code/analysis/evaluate_weights.py, unchanged) with
# 20,000 steps x 1024 walkers = 20,480,000 configurations, as for the authors'-code runs, into
# evaluation/attention_18e_eps3_seedSEED/.
# Safe to start again: train_moire.py continues from its newest checkpoint (and refuses a folder
# made with other settings); a finished run or evaluation is not repeated.
# Around the run only two passive monitors, every 10 s, in training/logs/.
cd "$(dirname "$0")" || exit 1
USAGE="usage: bash how_this_was_started.sh GPU SEED"
[ $# -eq 2 ] || { echo "$USAGE"; exit 2; }
case "$1" in 0|1) GPU=$1 ;; *) echo "$USAGE"; exit 2 ;; esac
[[ "$2" =~ ^[0-9]+$ ]] || { echo "$USAGE"; exit 2; }
SEED=$2
HERE=$PWD
CODE=$HOME/moire-nnvmc/learn/stage08_own_code
PY=${PY:-$HOME/moire-nnvmc/.venv/bin/python}
RUN=$HERE/training/attention_18e_eps3_seed$SEED
LOGS=$HERE/training/logs
mkdir -p $LOGS

stamp=$(date +%F_%H%M%S)
nvidia-smi -i $GPU --format=csv -l 10 \
    --query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,clocks.sm,fan.speed,clocks_event_reasons.active \
    >> $LOGS/gpu${GPU}_monitor_$stamp.csv 2>&1 &
gpu_monitor=$!
$PY $CODE/system_monitor.py --parent $$ --every 10 --out $LOGS/gpu${GPU}_system_$stamp.csv &
system_monitor=$!

echo "=== $(date '+%F %T')  start attention_18e_eps3_seed$SEED on GPU $GPU"
(cd $CODE && CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 $PY train_moire.py --network attention --eps 3 \
    --electrons 18 --settings released --laplacian folx --determinants slogdet --gradient real \
    --seed $SEED --out $RUN)
status=$?                                          # saved first: $(date) below would overwrite $?
echo "=== $(date '+%F %T')  attention_18e_eps3_seed$SEED ended with exit code $status"
if [ $status -eq 0 ]; then
    echo "=== $(date '+%F %T')  frozen-weight evaluation of attention_18e_eps3_seed$SEED on GPU $GPU"
    (cd $CODE && CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 $PY analysis/evaluate_weights.py $RUN --steps 20000)
    status=$?
    echo "=== $(date '+%F %T')  evaluation of attention_18e_eps3_seed$SEED ended with exit code $status"
fi
kill $gpu_monitor $system_monitor
