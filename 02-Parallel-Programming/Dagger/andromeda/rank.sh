#!/bin/bash
# One MPI rank started by andromeda/mpirun.jl: keep only this rank's share of
# the job's GPUs visible, then run the command.
set -euo pipefail
rank="${PMI_RANK:-${MPI_LOCALRANKID:-0}}"
ranks="${SC26_RANKS:-1}"
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=, read -ra devices <<< "$CUDA_VISIBLE_DEVICES"
    share=$(( ${#devices[@]} / ranks ))
    mine=("${devices[@]:$(( rank * share )):$share}")
    CUDA_VISIBLE_DEVICES="$(IFS=,; echo "${mine[*]}")"
    export CUDA_VISIBLE_DEVICES
fi
exec "$@"
