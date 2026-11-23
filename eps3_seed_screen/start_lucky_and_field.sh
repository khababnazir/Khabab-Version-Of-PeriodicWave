#!/bin/bash
# OUR code (learn/stage08_own_code/train_moire.py, unchanged) at eps = 3, the authors' released
# settings, the self-attention network: two kinds of short runs, judged early like the screen.
#
#   bash start_lucky_and_field.sh GPU lucky FIRST LAST
#       The authors' LUCKY START: the parameters and walkers their train.py made for seed 42
#       (capture_authors_start.py, captured once on this GPU, into authors_seed42_start_gpu.pkl),
#       given to our code as its iteration-0 checkpoint (lucky_start.py); our burn-in, Metropolis
#       moves and KFAC from seeds FIRST..LAST; 5,000 iterations each, in lucky_start/seed_S/.
#   bash start_lucky_and_field.sh GPU field SEED
#       Our code from seed SEED with the honeycomb field (field_training.py, lambda = 0.5) for
#       iterations 0 to 4,999, then the field off (plain train_moire.py, same folder) to 14,999,
#       in field/seed_SEED/.
#
# Start, from this folder (no flock around it: train_moire.py takes the GPU lock itself; the short
# capture takes it with flock):
#   setsid nohup bash start_lucky_and_field.sh 1 field 42 >> logs/gpu1_lucky_and_field.log 2>&1 < /dev/null &
# Safe to start again: train_moire.py continues every run from its newest checkpoint, a finished
# one ends at once, and a start or capture already written is not written again.
cd "$(dirname "$0")" || exit 1
USAGE="usage: bash start_lucky_and_field.sh GPU lucky FIRST LAST | GPU field SEED"
case "$1" in 0|1) GPU=$1 ;; *) echo "$USAGE"; exit 2 ;; esac
case "$2" in
    lucky) [ $# -eq 4 ] && [[ "$3" =~ ^[0-9]+$ && "$4" =~ ^[0-9]+$ && $3 -le $4 ]] || { echo "$USAGE"; exit 2; } ;;
    field) [ $# -eq 3 ] && [[ "$3" =~ ^[0-9]+$ ]] || { echo "$USAGE"; exit 2; } ;;
    *) echo "$USAGE"; exit 2 ;;
esac
HERE=$PWD
REPO=$HOME/moire-nnvmc
PY=${PY:-$REPO/.venv/bin/python}
LOCKS=${LOCKS:-$REPO/runs/gpu_locks}
OPTIONS="--network attention --eps 3 --electrons 18 --settings released --laplacian folx --determinants slogdet --gradient real"
mkdir -p logs lucky_start field

stamp=$(date +%F_%H%M%S)
nvidia-smi -i $GPU --format=csv -l 10 \
    --query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,clocks.sm,fan.speed,clocks_event_reasons.active \
    >> logs/gpu${GPU}_monitor_$stamp.csv 2>&1 &
gpu_monitor=$!
$PY $REPO/learn/stage08_own_code/system_monitor.py --parent $$ --every 10 --out logs/gpu${GPU}_system_$stamp.csv &
system_monitor=$!

ours () {   # label, log name, then the program and its options
    label=$1; log=$2; shift 2
    echo "=== $(date '+%F %T')  start $label on GPU $GPU"
    CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 $PY "$@" >> logs/$log.log 2>&1
    status=$?                                      # saved first: $(date) below would overwrite $?
    echo "=== $(date '+%F %T')  $label ended with exit code $status"
    return $status
}

if [ "$2" = lucky ]; then
    START=authors_seed42_start_gpu.pkl
    if [ ! -f $START ]; then
        echo "=== $(date '+%F %T')  capturing the authors' seed-42 start on GPU $GPU"
        CUDA_VISIBLE_DEVICES=$GPU flock -n $LOCKS/gpu$GPU.lock $PY capture_authors_start.py $START.tmp > logs/capture_gpu.log 2>&1 \
            && mv $START.tmp $START || { echo "=== capture failed (logs/capture_gpu.log)"; kill $gpu_monitor $system_monitor; exit 1; }
        JAX_PLATFORMS=cpu $PY -c "
import pickle, numpy as np, jax
a, b = (pickle.load(open(f, 'rb')) for f in ('$START', 'authors_seed42_start.pkl'))
print('GPU capture against the CPU one: backend', a['backend'], '| walkers max |diff|', float(np.abs(a['positions'] - b['positions']).max()),
      '| parameters max |diff|', max(float(np.abs(x - y).max()) for x, y in zip(jax.tree.leaves(a['params']), jax.tree.leaves(b['params']))))
" 2>&1 | grep -v 'cuInit\|xla_bridge\|jax_plugins\|plugin_module\|~~~\|cuda_versions\|local_device_count\|RuntimeError: jaxlib\|Traceback'
    fi
    for seed in $(seq $3 $4); do
        folder=lucky_start/seed_$seed
        if ! ls $folder/checkpoint_*.npz > /dev/null 2>&1; then
            JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= $PY lucky_start.py $START $folder $seed > logs/lucky_start_write_seed_$seed.log 2>&1 \
                || { echo "=== writing the start of $folder failed (logs/lucky_start_write_seed_$seed.log)"; continue; }
        fi
        ours $folder lucky_start_seed_$seed $REPO/learn/stage08_own_code/train_moire.py $OPTIONS --seed $seed --iterations 5000 --out "$HERE/$folder"
        [ $? -eq 75 ] && { echo "=== GPU $GPU or $folder is busy; stopping"; break; }
    done
else
    folder=field/seed_$3
    ours "$folder (field on, 0 to 4,999)" field_seed_$3_on field_training.py 0.5 $OPTIONS --seed $3 --iterations 5000 --out "$HERE/$folder" \
        && ours "$folder (field off, to 14,999)" field_seed_$3_off $REPO/learn/stage08_own_code/train_moire.py $OPTIONS --seed $3 --iterations 15000 --out "$HERE/$folder"
fi
kill $gpu_monitor $system_monitor
