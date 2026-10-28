#!/bin/bash
# Check the launcher's queue logic without training anything:  bash test_launcher.sh
# Exit status 0 only if every case behaves as expected.
#
# Stand-in programs replace Python and exit 0 (finished), 3 (failed) or
# 75 (folder already being trained).  Each queue has two runs.  Expected:
#   75  ->  the queue stops: 1 run started, the second NOT started on the busy GPU
#   3   ->  FAILED, then the independent second run: 2 started
#   0   ->  both runs finish: 2 started
#   and a queue for a GPU whose lock is already held starts nothing: 0 started.
W=$(mktemp -d)
mkdir -p $W/runs/gpu_locks $W/runs/training
for code in 0 3 75; do
    printf '#!/bin/bash\necho "      (stand-in ran: $*)"\nexit %s\n' $code > $W/fake$code
    chmod +x $W/fake$code
done
source "$(dirname "$0")/launch_fig5.sh" --functions-only
RUNS=$W/runs
LOCKS=$W/runs/gpu_locks
failures=0

check () {   # description, expected number of runs started, queue output
    started=$(printf '%s\n' "$3" | grep -c "stand-in ran")
    printf '%s\n' "$3"
    if [ "$started" -eq "$2" ]; then
        echo "ok    $1: $started started"
    else
        echo "FAIL  $1: expected $2 started, got $started"
        failures=$((failures + 1))
    fi
}

for case in "75 1" "3 2" "0 2"; do
    set -- $case
    PY=$W/fake$1
    # $( ) runs the queue in a subshell, which releases its GPU lock when it ends.
    check "stand-in exits $1" $2 "$(queue 0 CustomPsiformer 5 first_run  SlaterNet 5 second_run 2>&1)"
done

( flock $W/runs/gpu_locks/gpu0.lock sleep 4 ) &
sleep 1
PY=$W/fake0
check "GPU 0 lock already held" 0 "$(queue 0 CustomPsiformer 5 first_run  SlaterNet 5 second_run 2>&1)"
wait
rm -rf $W
[ $failures -eq 0 ] && echo "all launcher checks passed" || echo "$failures launcher check(s) FAILED"
exit $failures
