#!/bin/bash
# A fresh run of our code (learn/stage08_own_code/train_moire.py, unchanged), eps = 3, 18 electrons
# on 27 sites, the authors' released settings, 200,000 iterations, then its frozen-weight
# evaluation (learn/stage08_own_code/analysis/evaluate_weights.py, unchanged; 20,000 steps x 1024
# walkers = 20,480,000 configurations) into long/evaluation/NAME/.
#
#   bash train_to_200000.sh GPU slaternet|attention SEED      ->  long/<network>_seed_<SEED>/
#
# Start, from this folder (train_moire.py takes the GPU lock itself):
#   setsid nohup bash train_to_200000.sh 1 slaternet 36 >> logs/long_slaternet_seed_36.log 2>&1 < /dev/null &
# Safe to start again: train_moire.py continues from its newest checkpoint; a finished run or
# evaluation is not repeated.
cd "$(dirname "$0")" || exit 1
USAGE="usage: bash train_to_200000.sh GPU slaternet|attention SEED"
[ $# -eq 3 ] || { echo "$USAGE"; exit 2; }
case "$1" in 0|1) GPU=$1 ;; *) echo "$USAGE"; exit 2 ;; esac
case "$2" in slaternet|attention) NETWORK=$2 ;; *) echo "$USAGE"; exit 2 ;; esac
[[ "$3" =~ ^[0-9]+$ ]] || { echo "$USAGE"; exit 2; }
SEED=$3
NAME=${NETWORK}_seed_$SEED
HERE=$PWD
REPO=$HOME/moire-nnvmc
PY=${PY:-$REPO/.venv/bin/python}
CODE=$REPO/learn/stage08_own_code
RUN=$HERE/long/$NAME
mkdir -p long/evaluation logs

stamp=$(date +%F_%H%M%S)
nvidia-smi -i $GPU --format=csv -l 10 \
    --query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,clocks.sm,fan.speed,clocks_event_reasons.active \
    >> logs/long_gpu${GPU}_monitor_$stamp.csv 2>&1 &
gpu_monitor=$!
$PY $CODE/system_monitor.py --parent $$ --every 10 --out logs/long_gpu${GPU}_system_$stamp.csv &
system_monitor=$!

echo "=== $(date '+%F %T')  start long/$NAME on GPU $GPU"
(cd $CODE && CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 $PY train_moire.py --network $NETWORK --eps 3 --electrons 18 \
    --settings released --laplacian folx --determinants slogdet --gradient real --seed $SEED --iterations 200000 --out "$RUN")
status=$?                                          # saved first: $(date) below would overwrite $?
echo "=== $(date '+%F %T')  long/$NAME ended with exit code $status"
if [ $status -eq 0 ]; then
    echo "=== $(date '+%F %T')  frozen-weight evaluation of long/$NAME on GPU $GPU"
    (cd $CODE && CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 $PY analysis/evaluate_weights.py "$RUN" --steps 20000 \
        --out "$HERE/long/evaluation/$NAME")
    status=$?
    echo "=== $(date '+%F %T')  evaluation of long/$NAME ended with exit code $status"
fi
kill $gpu_monitor $system_monitor
