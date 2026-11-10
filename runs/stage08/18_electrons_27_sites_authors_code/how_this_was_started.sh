#!/bin/bash
# 2026-11-10.  The authors' released code, exactly as published (external/PeriodicWave at commit
# 6525d2f3eec53f085b55d485e1bff45d61ef1782, no file changed), started with THEIR script and THEIR
# arguments:   spins  sites  m*/m_e  1/eps  a_M (nm)  V (meV)  phi (degrees)  network
# 18 electrons on 27 sites, the self-attention network (CustomPsiformer), eps = 5 and then eps = 10.
# No line of ours runs inside the training: no wrapper, no changed setting, their random seed.
# Around it, only: one GPU made visible, the output kept in logs/, and two passive monitors
# (nvidia-smi and system_monitor.py, every 10 s).  Started with
#   setsid nohup flock -n ~/moire-nnvmc/runs/gpu_locks/gpu1.lock bash how_this_was_started.sh >> logs/gpu1.log 2>&1 &
# (flock holds the lock our own tools use, so that nothing else of ours starts on this GPU).
# Their script writes into results/2deg-CoulombMoire/CustomPsiformer/<parameters>/ below this folder,
# and continues from its last checkpoint if this is started again.
cd "$(dirname "$0")" || exit 1
REPO=$HOME/moire-nnvmc
PY=$REPO/.venv/bin/python
SCRIPT=$REPO/external/PeriodicWave/periodicwave/configs/run_2deg_triangular_potential.py
GPU=1
stamp=$(date +%F_%H%M%S)
nvidia-smi -i $GPU --format=csv -l 10 \
    --query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,clocks.sm,fan.speed,clocks_event_reasons.active \
    >> logs/gpu${GPU}_monitor_$stamp.csv 2>&1 &
gpu_monitor=$!
$PY $REPO/learn/stage08_own_code/system_monitor.py --parent $$ --every 10 --out logs/gpu${GPU}_system_$stamp.csv &
system_monitor=$!
for inverse_eps in 0.2 0.1; do          # eps = 5, then eps = 10
    echo "=== $(date '+%F %T')  start 1/eps = $inverse_eps on GPU $GPU"
    CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 $PY $SCRIPT 18_0 27 0.35 $inverse_eps 8.031 15 45 CustomPsiformer
    echo "=== $(date '+%F %T')  1/eps = $inverse_eps ended with exit code $?"
done
kill $gpu_monitor $system_monitor
