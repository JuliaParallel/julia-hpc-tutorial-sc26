#!/bin/bash
# Compile the packages and check the GPUs and MPI (frontier/setup.sh) in a
# short job of its own, after prepare.sh on the login node. The web console
# runs this on Connect, so its runs start from compiled packages:
#   bash frontier/precompile.sh
set -euo pipefail
source "$(dirname "$0")/env.sh"
args=(-A "$SC26_ACCOUNT" -p "${SC26_PARTITION:-batch}" -N 1 -n 1 -c 7 --threads-per-core=1
      --gpus-per-task=8 --gpu-bind=none -t "${SC26_PRECOMPILE_TIME:-00:30:00}"
      -J "${SC26_SESSION:-sc26-dagger-precompile}")
if [[ -n "${SC26_RESERVATION:-}" ]]; then args+=(--reservation="$SC26_RESERVATION"); fi
exec srun "${args[@]}" bash "$SC26_ROOT/frontier/setup.sh"
