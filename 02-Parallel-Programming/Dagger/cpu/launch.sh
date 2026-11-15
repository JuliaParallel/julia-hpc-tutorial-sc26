#!/usr/bin/env bash
# One local process group per run; MPI ranks share this VM's CPUs.
set -euo pipefail
source "$(dirname "$0")/env.sh"
ranks="$1"
shift
export JULIA_PKG_OFFLINE=true
if [[ "${SC26_CPU_MPI:-0}" == 1 ]]; then
    exec julia --startup-file=no --project="$JULIA_PROJECT" -e '
        using MPI
        ranks = parse(Int, popfirst!(ARGS))
        cmd = `$(MPI.mpiexec()) -n $ranks $(Base.julia_cmd()) --project=$(Base.active_project()) $ARGS`
        run(cmd)
    ' "$ranks" "$@"
else
    exec julia --project="$JULIA_PROJECT" "$@"
fi
