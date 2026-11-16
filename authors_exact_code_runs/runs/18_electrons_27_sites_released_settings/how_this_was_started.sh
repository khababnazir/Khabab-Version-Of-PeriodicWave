#!/bin/bash
# The authors' released code, exactly as published (external/PeriodicWave at commit
# 6525d2f3eec53f085b55d485e1bff45d61ef1782, no file changed), started with THEIR script and THEIR
# arguments:   spins  sites  m*/m_e  1/eps  a_M (nm)  V (meV)  phi (degrees)  network
# 18 electrons on 27 sites, the self-attention network (CustomPsiformer), eps = 5 and then eps = 10.
# No line of ours runs inside the training: no wrapper, no changed setting, their random seed.
#
#   cd runs/stage08/18_electrons_27_sites_authors_code
#   setsid nohup flock -n ~/moire-nnvmc/runs/gpu_locks/gpuG.lock bash how_this_was_started.sh G [0.1 | 0.2] >> logs/gpuG.log 2>&1 < /dev/null &
# with G = 0 or 1 (flock holds the lock our own tools use on that GPU).  Without the second
# argument: eps = 5 and then eps = 10; with it (added 2026-11-12) only that 1/eps.
#
# Around the training, only: one GPU made visible, the output kept in logs/, two passive monitors
# every 10 s (nvidia-smi, system_monitor.py), and -- second version, 2026-11-10, after GPU 1 fell
# off the bus 11 minutes into the first start and the eps = 10 run began on the CPU without a
# word -- a check before each run that the GPU can start a program at all.  Their script writes
# into results/2deg-CoulombMoire/CustomPsiformer/<parameters>/ below this folder and continues
# from its last checkpoint if this is started again.
cd "$(dirname "$0")" || exit 1
USAGE="usage: bash how_this_was_started.sh GPU [INVERSE_EPS]    (GPU: 0 or 1;  1/eps: 0.1 or 0.2, default both)"
case "$1" in 0|1) GPU=$1 ;; *) echo "$USAGE"; exit 2 ;; esac
case "${2:-both}" in both) EPS_LIST="0.2 0.1" ;; 0.1|0.2) EPS_LIST=$2 ;; *) echo "$USAGE"; exit 2 ;; esac
[ $# -le 2 ] || { echo "$USAGE"; exit 2; }
REPO=$HOME/moire-nnvmc
PY=${PY:-$REPO/.venv/bin/python}
mkdir -p logs

gpu_can_start_a_program () {    # asks the driver only (cuInit); creates nothing on the GPU
    CUDA_VISIBLE_DEVICES=$GPU $REPO/.venv/bin/python -c \
        "import ctypes, sys; sys.exit(0 if ctypes.CDLL('libcuda.so.1').cuInit(0) == 0 else 1)"
}

if ! gpu_can_start_a_program; then
    echo "=== $(date '+%F %T')  GPU $GPU cannot start a program (the driver refuses; reboot needed?). Nothing started."
    exit 1
fi
stamp=$(date +%F_%H%M%S)
nvidia-smi -i $GPU --format=csv -l 10 \
    --query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,clocks.sm,fan.speed,clocks_event_reasons.active \
    >> logs/gpu${GPU}_monitor_$stamp.csv 2>&1 &
gpu_monitor=$!
$REPO/.venv/bin/python $REPO/learn/stage08_own_code/system_monitor.py --parent $$ --every 10 --out logs/gpu${GPU}_system_$stamp.csv &
system_monitor=$!
SCRIPT=$REPO/external/PeriodicWave/periodicwave/configs/run_2deg_triangular_potential.py
for inverse_eps in $EPS_LIST; do         # eps = 5, then eps = 10, or the one asked for
    if ! gpu_can_start_a_program; then
        echo "=== $(date '+%F %T')  GPU $GPU cannot start a program any more; the queue stops before 1/eps = $inverse_eps"
        break
    fi
    echo "=== $(date '+%F %T')  start 1/eps = $inverse_eps on GPU $GPU"
    CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 $PY $SCRIPT 18_0 27 0.35 $inverse_eps 8.031 15 45 CustomPsiformer
    status=$?    # saved first: $(date) below would overwrite $?
    echo "=== $(date '+%F %T')  1/eps = $inverse_eps ended with exit code $status"
done
kill $gpu_monitor $system_monitor
