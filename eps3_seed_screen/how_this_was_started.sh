#!/bin/bash
# eps = 3, 18 electrons on 27 sites, the self-attention network, the authors' RELEASED settings:
# many short random starts of BOTH codes, 5,000 iterations each, to see how often a start ends in
# the honeycomb (in the authors' code, seed 42 had it by its first checkpoint, 4,709).
#
#   bash how_this_was_started.sh GPU authors|ours FIRST LAST       seeds FIRST to LAST, one after another
#
# authors: their script with the seed set and 5,000 iterations (run_2deg_triangular_potential_screen.py;
#          changes.diff), run in authors/seed_S/ (their output in results/... there);
# ours:    learn/stage08_own_code/train_moire.py, unchanged: --network attention --eps 3 --electrons 18
#          --settings released --laplacian folx --determinants slogdet --gradient real --seed S
#          --iterations 5000 --out ours/seed_S.
# Start, from this folder (train_moire.py takes the GPU lock itself, so the "ours" queue must not):
#   setsid nohup flock -n ~/moire-nnvmc/runs/gpu_locks/gpu0.lock bash how_this_was_started.sh 0 authors 1 12 >> logs/gpu0.log 2>&1 < /dev/null &
#   setsid nohup bash how_this_was_started.sh 1 ours 1 12 >> logs/gpu1.log 2>&1 < /dev/null &
# Safe to start again: a seed whose final checkpoint exists is skipped (authors) or ends at once (ours).
# Around the runs only two passive monitors, every 10 s, in logs/.
cd "$(dirname "$0")" || exit 1
USAGE="usage: bash how_this_was_started.sh GPU authors|ours FIRST LAST"
[ $# -eq 4 ] || { echo "$USAGE"; exit 2; }
case "$1" in 0|1) GPU=$1 ;; *) echo "$USAGE"; exit 2 ;; esac
case "$2" in authors|ours) CODE=$2 ;; *) echo "$USAGE"; exit 2 ;; esac
[[ "$3" =~ ^[0-9]+$ && "$4" =~ ^[0-9]+$ && $3 -le $4 ]] || { echo "$USAGE"; exit 2; }
HERE=$PWD
REPO=$HOME/moire-nnvmc
PY=${PY:-$REPO/.venv/bin/python}
ITERATIONS=5000
ARGS="18_0 27 0.35 0.3333333333333333 8.031 15 45 CustomPsiformer"
OUT=results/2deg-CoulombMoire/CustomPsiformer/el18_0_N27_V4.44370908_45.0_U17.70578894
mkdir -p logs authors ours

stamp=$(date +%F_%H%M%S)
nvidia-smi -i $GPU --format=csv -l 10 \
    --query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,clocks.sm,fan.speed,clocks_event_reasons.active \
    >> logs/gpu${GPU}_monitor_$stamp.csv 2>&1 &
gpu_monitor=$!
$PY $REPO/learn/stage08_own_code/system_monitor.py --parent $$ --every 10 --out logs/gpu${GPU}_system_$stamp.csv &
system_monitor=$!

for seed in $(seq $3 $4); do
    if [ $CODE = authors ]; then
        folder=authors/seed_$seed
        if [ -f $folder/$OUT/qmcjax_ckpt_$(printf %06d $((ITERATIONS - 1))).npz ]; then
            echo "=== $(date '+%F %T')  $folder finished before; skipped"
            continue
        fi
        mkdir -p $folder
        echo "=== $(date '+%F %T')  start $folder on GPU $GPU"
        (cd $folder && SCREEN_SEED=$seed SCREEN_ITERATIONS=$ITERATIONS CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 \
            $PY "$HERE/run_2deg_triangular_potential_screen.py" $ARGS > run.log 2>&1)
    else
        folder=ours/seed_$seed
        echo "=== $(date '+%F %T')  start $folder on GPU $GPU"
        (cd $REPO/learn/stage08_own_code && CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 $PY train_moire.py \
            --network attention --eps 3 --electrons 18 --settings released --laplacian folx --determinants slogdet \
            --gradient real --seed $seed --iterations $ITERATIONS --out "$HERE/$folder" > "$HERE/logs/ours_seed_$seed.log" 2>&1)
    fi
    status=$?                                      # saved first: $(date) below would overwrite $?
    echo "=== $(date '+%F %T')  $folder ended with exit code $status"
    if [ $status -eq 75 ]; then
        echo "=== GPU $GPU or $folder is busy; this queue stops"
        break
    fi
done
kill $gpu_monitor $system_monitor
