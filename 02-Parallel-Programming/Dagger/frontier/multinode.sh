#!/bin/bash
# "Multi-node" programming model, two MPI ranks on the existing workshop node.
# Odo tutorial capacity limits the launch layout; Julia code also runs across nodes.
# bash frontier/multinode.sh cpu   (or gpu for two distinct GPUs per rank)
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
mode="${1:-}"
case "$mode" in
    cpu) demo=06_multinode.jl; gpus=0 ;;
    gpu) demo=07_multinode_multigpu.jl; gpus="${SC26_GPUS_PER_TASK:-2}" ;;
    *) echo "Usage: multinode.sh cpu|gpu [demo options]" >&2; exit 2 ;;
esac
shift
: "${SLURM_JOB_ID:?Obtain an allocation with frontier/allocate.sh first}"
export SC26_NODES=1 SC26_GPUS_PER_TASK="$gpus" JULIA_PKG_OFFLINE=true
exec bash "$root/frontier/launch.sh" 2 "$root/demos/$demo" "$@"
