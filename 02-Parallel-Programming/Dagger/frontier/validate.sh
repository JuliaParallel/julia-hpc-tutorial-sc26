#!/bin/bash
# Run after prepare.sh + setup.sh, from the login node with an allocation.
# SLURM_JOB_ID=123 bash frontier/validate.sh
# Course checks all fit in one node, including four MPI GPU ranks.
set -uo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"
logdir="${SC26_VALIDATION_LOGS:-$root/results/validation}"
mkdir -p "$logdir"
failures=0
check() {
    local name="$1"; shift
    echo "Checking $name..."
    if "$@" > "$logdir/$name.log" 2>&1; then
        echo "PASS $name"
    else
        echo "FAIL $name (see $logdir/$name.log)"
        tail -15 "$logdir/$name.log"
        failures=$((failures + 1))
    fi
}
export SC26_NODES=1
export SC26_STEP_TIME="${SC26_STEP_TIME:-00:05:00}"
check tasks bash frontier/launch.sh 1 demos/01_tasks.jl
check datadeps bash frontier/launch.sh 1 demos/02_datadeps.jl
check gpu-demo bash frontier/launch.sh 1 demos/03_gpu.jl
check multigpu-demo bash frontier/launch.sh 1 demos/05_multigpu.jl
small=(--impl=solution --sizes=1024 --bs=256 --trials=1)
check cpu bash frontier/run.sh 1 "${small[@]}"
echo "Checking the incomplete template's diagnostic..."
if bash frontier/run.sh 1 --impl=student --sizes=1024 --bs=256 --trials=1 > "$logdir/student-template.log" 2>&1; then
    echo "PASS student implementation (template already completed)"
elif grep -q 'FAIL: your code threw an error during warm-up' "$logdir/student-template.log"; then
    echo "PASS incomplete-template diagnostic"
else
    echo "FAIL unexpected student-template error"
    failures=$((failures + 1))
fi
check gpu bash frontier/run.sh 1 "${small[@]}" --target=gpu
check multigpu bash frontier/run.sh 1 "${small[@]}" --target=gpu --gpus=all
check mixed bash frontier/run.sh 1 "${small[@]}" --target=mixed
check mpi-demo bash frontier/launch.sh 2 demos/04_mpi.jl
check mpi-cpu bash frontier/run.sh 2 "${small[@]}" --mpi --strict
check mpi-gpu bash frontier/run.sh 2 "${small[@]}" --mpi --strict --target=gpu
check mpi-four-rank-demo bash frontier/launch.sh 4 demos/05_multigpu.jl --mpi
check mpi-four-rank-cpu bash frontier/run.sh 4 "${small[@]}" --mpi --strict
check mpi-four-rank-gpu bash frontier/run.sh 4 "${small[@]}" --mpi --strict --target=gpu
echo "$failures failed checks; logs: $logdir"
(( failures == 0 ))
