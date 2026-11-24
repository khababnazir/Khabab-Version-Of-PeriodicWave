#!/bin/bash
# Continue one start of the screen to 200,000 iterations, then its frozen-weight evaluation.
# The screen folder is left as it is: the start is COPIED (settings.json, train_stats.csv and its
# checkpoints) into long/NAME/, and our train_moire.py (unchanged) continues the copy with the same
# options and --iterations 200000 (it accepts a larger target for the same experiment and continues
# from the newest checkpoint).  Then learn/stage08_own_code/analysis/evaluate_weights.py (unchanged),
# 20,000 steps x 1024 walkers = 20,480,000 configurations, into long/evaluation/NAME/.
#
#   bash continue_to_200000.sh GPU SOURCE NAME      e.g.  0 ours/seed_36 ours_seed_36
#
# Start, from this folder (train_moire.py takes the GPU lock itself):
#   setsid nohup bash continue_to_200000.sh 0 ours/seed_36 ours_seed_36 >> logs/long_ours_seed_36.log 2>&1 < /dev/null &
# Safe to start again: the copy is made only once; train_moire.py continues from the newest
# checkpoint; a finished evaluation is not repeated.
cd "$(dirname "$0")" || exit 1
USAGE="usage: bash continue_to_200000.sh GPU SOURCE NAME"
[ $# -eq 3 ] || { echo "$USAGE"; exit 2; }
case "$1" in 0|1) GPU=$1 ;; *) echo "$USAGE"; exit 2 ;; esac
SOURCE=$2; NAME=$3
[ -f "$SOURCE/settings.json" ] || { echo "$SOURCE: no settings.json"; exit 2; }
HERE=$PWD
REPO=$HOME/moire-nnvmc
PY=${PY:-$REPO/.venv/bin/python}
CODE=$REPO/learn/stage08_own_code
RUN=$HERE/long/$NAME
mkdir -p long/evaluation logs
if [ ! -d "$RUN" ]; then
    mkdir -p "$RUN.copying" && cp -p "$SOURCE"/settings.json "$SOURCE"/train_stats.csv "$SOURCE"/checkpoint_*.npz "$RUN.copying"/ \
        && cp -p "$SOURCE"/starts.log "$RUN.copying"/ 2>/dev/null; mv "$RUN.copying" "$RUN"
    echo "=== $(date '+%F %T')  copied $SOURCE to long/$NAME"
fi
SEED=$($PY -c "import json, sys; print(json.load(open(sys.argv[1]))['seed'])" "$RUN/settings.json")

stamp=$(date +%F_%H%M%S)
nvidia-smi -i $GPU --format=csv -l 10 \
    --query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,clocks.sm,fan.speed,clocks_event_reasons.active \
    >> logs/long_gpu${GPU}_monitor_$stamp.csv 2>&1 &
gpu_monitor=$!
$PY $CODE/system_monitor.py --parent $$ --every 10 --out logs/long_gpu${GPU}_system_$stamp.csv &
system_monitor=$!

echo "=== $(date '+%F %T')  continue long/$NAME (seed $SEED) to 200,000 on GPU $GPU"
(cd $CODE && CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 $PY train_moire.py --network attention --eps 3 --electrons 18 \
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
