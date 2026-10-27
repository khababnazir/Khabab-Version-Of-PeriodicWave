#!/bin/bash
# Check the launcher's queue logic without training anything:  bash test_launcher.sh
#
# Stand-in programs replace Python and exit 0 (finished), 3 (failed) or
# 75 (folder already being trained).  Expected:
#   75  ->  the queue stops; its second run is NOT started on the busy GPU
#   3   ->  FAILED (exit code 3), then the independent second run
#   0   ->  both runs finish
#   and a second queue for a GPU whose queue is running stops at once.
W=$(mktemp -d)
mkdir -p $W/runs/gpu_locks $W/runs/training
for code in 0 3 75; do
    printf '#!/bin/bash\necho "      (stand-in ran: $*)"\nexit %s\n' $code > $W/fake$code
    chmod +x $W/fake$code
done
source "$(dirname "$0")/launch_fig5.sh" --functions-only
RUNS=$W/runs
for code in 75 3 0; do
    echo "== stand-in exits $code:"
    PY=$W/fake$code
    queue 0 CustomPsiformer 5 first_run  SlaterNet 5 second_run
    exec 9>&-      # release this test shell's GPU lock; a real queue releases it by exiting
done
echo "== a queue for GPU 0 already holds its lock:"
( flock $W/runs/gpu_locks/gpu0.lock sleep 4 ) &
sleep 1
PY=$W/fake0
queue 0 CustomPsiformer 5 first_run  SlaterNet 5 second_run
exec 9>&-
wait
rm -rf $W
