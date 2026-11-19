#!/bin/bash
# eps = 3, the authors' code, their released settings, the self-attention network, FIXED SEED 42
# (two copies of their script in this folder, changes.diff).  Two runs that start from the same
# initial weights and walkers:
#
#   bash how_this_was_started.sh G ordinary   ordinary training, 200,000 iterations, in ordinary/
#   bash how_this_was_started.sh G field      the honeycomb test, in honeycomb_field/:
#       1. iterations 0 to 19,999 with their Hamiltonian + the honeycomb field (honeycomb_field.py),
#          in phase1_field_on/; their code saves a checkpoint at iteration 19,999;
#       2. S(kappa) and the sublattice shares of that checkpoint (check_honeycomb.py, into
#          honeycomb_field/phase1_check.csv and the log);
#       3. that checkpoint copied into phase2_field_off/, where their seed-42 script (no field, the
#          same Hamiltonian as the ordinary run) continues from it, iterations 20,000 to 199,999.
#
# Start (G = 0 or 1), from this folder:
#   setsid nohup flock -n ~/moire-nnvmc/runs/gpu_locks/gpuG.lock bash how_this_was_started.sh G MODE >> logs/gpuG.log 2>&1 < /dev/null &
# Safe to start again: their code continues from its newest checkpoint; a finished phase 1 is not
# repeated and its checkpoint is copied only into an empty phase-2 folder.
# Around the training, only: one GPU visible, two passive monitors every 10 s (logs/), and a check
# that the GPU can start a program.
cd "$(dirname "$0")" || exit 1
USAGE="usage: bash how_this_was_started.sh GPU ordinary|field"
case "$1" in 0|1) GPU=$1 ;; *) echo "$USAGE"; exit 2 ;; esac
case "$2" in ordinary|field) MODE=$2 ;; *) echo "$USAGE"; exit 2 ;; esac
[ $# -eq 2 ] || { echo "$USAGE"; exit 2; }
HERE=$PWD
REPO=$HOME/moire-nnvmc
PY=${PY:-$REPO/.venv/bin/python}
ARGS="18_0 27 0.35 0.3333333333333333 8.031 15 45 CustomPsiformer"     # 1/eps = 1/3, as the first eps = 3 run
OUT=results/2deg-CoulombMoire/CustomPsiformer/el18_0_N27_V4.44370908_45.0_U17.70578894
mkdir -p logs

gpu_can_start_a_program () {    # asks the driver only (cuInit); creates nothing on the GPU
    CUDA_VISIBLE_DEVICES=$GPU $REPO/.venv/bin/python -c \
        "import ctypes, sys; sys.exit(0 if ctypes.CDLL('libcuda.so.1').cuInit(0) == 0 else 1)"
}

train () {   # folder script: one training run of their code, in that folder
    mkdir -p "$1"
    if ! gpu_can_start_a_program; then
        echo "=== $(date '+%F %T')  GPU $GPU cannot start a program (the driver refuses; reboot needed?). Nothing started."
        return 1
    fi
    echo "=== $(date '+%F %T')  start $(basename $2) in $1 on GPU $GPU"
    (cd "$1" && CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 $PY "$HERE/$2" $ARGS)
    status=$?    # saved first: $(date) below would overwrite $?
    echo "=== $(date '+%F %T')  $(basename $2) in $1 ended with exit code $status"
    return $status
}

stamp=$(date +%F_%H%M%S)
nvidia-smi -i $GPU --format=csv -l 10 \
    --query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,clocks.sm,fan.speed,clocks_event_reasons.active \
    >> logs/gpu${GPU}_monitor_$stamp.csv 2>&1 &
gpu_monitor=$!
$PY $REPO/learn/stage08_own_code/system_monitor.py --parent $$ --every 10 --out logs/gpu${GPU}_system_$stamp.csv &
system_monitor=$!

if [ "$MODE" = ordinary ]; then
    train ordinary run_2deg_triangular_potential_seed42.py
else
    P1=honeycomb_field/phase1_field_on
    P2=honeycomb_field/phase2_field_off
    LAST=qmcjax_ckpt_019999.npz
    if [ ! -f $P1/$OUT/$LAST ]; then
        train $P1 run_2deg_triangular_potential_seed42_field.py
    fi
    if [ -f $P1/$OUT/$LAST ]; then
        if [ ! -f honeycomb_field/phase1_check.csv ]; then
            echo "=== $(date '+%F %T')  did a honeycomb form? the walkers of $LAST:"
            JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= $HOME/.local/bin/uv run --no-project --python $PY --with matplotlib --with pyblock \
                python check_honeycomb.py $P1/$OUT/$LAST honeycomb_field/phase1_check.csv 2>&1 | grep -v 'cuInit\|xla_bridge\|jax_plugins\|plugin_module\|~~~\|cuda_versions\|local_device_count\|Traceback'
        fi
        mkdir -p $P2/$OUT
        if ! ls $P2/$OUT/qmcjax_ckpt_*.npz > /dev/null 2>&1; then
            cp -p $P1/$OUT/$LAST $P2/$OUT/
            echo "=== $(date '+%F %T')  copied $LAST into $P2 (the field is off from iteration 20,000)"
        fi
        train $P2 run_2deg_triangular_potential_seed42.py
    else
        echo "=== $(date '+%F %T')  phase 1 left no $LAST; phase 2 not started"
    fi
fi
kill $gpu_monitor $system_monitor
