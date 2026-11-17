#!/bin/bash
# The authors' released code, exactly as published (external/PeriodicWave at commit
# 6525d2f3eec53f085b55d485e1bff45d61ef1782, no file changed), started with THEIR script and THEIR
# released settings, at eps = 3:
#   spins  sites  m*/m_e  1/eps  a_M (nm)  V (meV)  phi (degrees)  network
#   18_0   27     0.35    1/3    8.031     15       45             CustomPsiformer | SlaterNet
# 18 electrons on 27 sites (the system of the paper's Table I and Fig. 6).  Asked for by Khabab's
# professor on 2026-11-17: the authors' code as it is, at eps = 3, both networks.
# No line of ours runs inside the training: no wrapper, no changed setting, their random seed.
#
#   cd authors_exact_code_runs/runs/18_electrons_27_sites_eps3_released_settings
#   setsid nohup flock -n ~/moire-nnvmc/runs/gpu_locks/gpuG.lock bash how_this_was_started.sh G NETWORK >> logs/gpuG.log 2>&1 < /dev/null &
# with G = 0 or 1 and NETWORK = CustomPsiformer (self-attention) or SlaterNet.
#
# Around the training, only (as in ../18_electrons_27_sites_released_settings/): one GPU made
# visible, the output kept in logs/, two passive monitors every 10 s (nvidia-smi,
# system_monitor.py), and a check before the run that the GPU can start a program at all.
# Their script writes into results/2deg-CoulombMoire/<NETWORK>/<parameters>/ below this folder
# and continues from its last checkpoint if this is started again.
cd "$(dirname "$0")" || exit 1
USAGE="usage: bash how_this_was_started.sh GPU NETWORK    (GPU: 0 or 1;  NETWORK: CustomPsiformer or SlaterNet)"
case "$1" in 0|1) GPU=$1 ;; *) echo "$USAGE"; exit 2 ;; esac
case "$2" in CustomPsiformer|SlaterNet) NETWORK=$2 ;; *) echo "$USAGE"; exit 2 ;; esac
[ $# -eq 2 ] || { echo "$USAGE"; exit 2; }
INVERSE_EPS=0.3333333333333333     # 1/3: their script reads float(argv[4]); this is the double nearest to 1/3
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
echo "=== $(date '+%F %T')  start $NETWORK, 1/eps = $INVERSE_EPS, on GPU $GPU"
CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 $PY $SCRIPT 18_0 27 0.35 $INVERSE_EPS 8.031 15 45 $NETWORK
status=$?    # saved first: $(date) below would overwrite $?
echo "=== $(date '+%F %T')  $NETWORK, 1/eps = $INVERSE_EPS, ended with exit code $status"
kill $gpu_monitor $system_monitor
