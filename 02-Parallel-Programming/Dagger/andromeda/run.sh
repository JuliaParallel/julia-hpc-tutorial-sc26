#!/bin/bash
# Harness wrapper, one Slurm job per call: run.sh RANKS GPUS [harness options].
#   bash andromeda/run.sh 1 0 --impl=student                   # CPU threads
#   bash andromeda/run.sh 1 1 --impl=student --target=gpu      # one GPU
#   bash andromeda/run.sh 2 0 --impl=student --mpi --strict    # two MPI ranks
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
np="${1:?Usage: run.sh RANKS GPUS [harness options]}"
gpus="${2:?Usage: run.sh RANKS GPUS [harness options]}"
shift 2
exec bash "$root/andromeda/launch.sh" "$np" "$gpus" "$root/harness/run.jl" "$@"
