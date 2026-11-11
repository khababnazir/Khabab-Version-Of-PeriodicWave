#!/bin/bash
# Waits until GPU 0 is free, then continues OUR paper-network run (eps = 5) from its newest
# checkpoint and restarts its watchdog.  Written 2026-11-11: the server was restarted at
# iteration 102,300 of 150,000 and GPU 0 was given to the authors'-code Table II run first.
#
#   cd ~/moire-nnvmc
#   setsid nohup bash runs/stage08/18_electrons_27_sites_paper_network_real_gradient/resume_when_gpu0_is_free.sh \
#       >> runs/stage08/18_electrons_27_sites_paper_network_real_gradient/training/logs/waiting_for_gpu0.log 2>&1 < /dev/null &
#
# It starts nothing if GPU 0 cannot start a program (a run would silently use the CPU).
# For a test: LOCK=<file> DRY=1 prints the commands instead of running them.
REPO=$HOME/moire-nnvmc
LOCK=${LOCK:-$REPO/runs/gpu_locks/gpu0.lock}
RUN=runs/stage08/18_electrons_27_sites_paper_network_real_gradient/training/attention_18e_eps5
run () { if [ -n "$DRY" ]; then echo "WOULD RUN: $*"; else "$@"; fi; }

cd $REPO || exit 1
echo "=== $(date '+%F %T')  waiting for $LOCK"
flock "$LOCK" true                      # returns when whoever holds GPU 0 has ended
echo "=== $(date '+%F %T')  GPU 0 is free"
if ! CUDA_VISIBLE_DEVICES=0 .venv/bin/python -c \
        "import ctypes, sys; sys.exit(0 if ctypes.CDLL('libcuda.so.1').cuInit(0) == 0 else 1)"; then
    echo "=== $(date '+%F %T')  GPU 0 cannot start a program (reboot needed?). Nothing started."
    exit 1
fi
run bash learn/stage08_own_code/launch_table1.sh paper5 150000
[ -z "$DRY" ] && sleep 300              # the watchdog stops if it sees an idle run twice
if [ -n "$DRY" ]; then
    echo "WOULD RUN: watch_run.py $RUN --every 1800"
else
    setsid nohup .venv/bin/python learn/stage08_own_code/watch_run.py $RUN --every 1800 \
        >> $RUN/../logs/watch.log 2>&1 < /dev/null &
fi
echo "=== $(date '+%F %T')  done"
