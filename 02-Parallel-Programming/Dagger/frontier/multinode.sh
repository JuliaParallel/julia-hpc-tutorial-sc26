#!/bin/bash
# Presenter/optional demos: one MPI rank per node, across two physical nodes.
# Obtain resources on the login node: SC26_NODES=2 bash frontier/allocate.sh
# Then: bash frontier/multinode.sh cpu   (or gpu for two GPUs per node)
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
mode="${1:-}"
case "$mode" in
    cpu) demo=06_multinode.jl; gpus=0 ;;
    gpu) demo=07_multinode_multigpu.jl; gpus="${SC26_GPUS_PER_TASK:-2}" ;;
    *) echo "Usage: multinode.sh cpu|gpu [demo options]" >&2; exit 2 ;;
esac
shift
: "${SLURM_JOB_ID:?Obtain a two-node allocation with SC26_NODES=2 bash frontier/allocate.sh first}"
nodes="${SLURM_JOB_NUM_NODES:-${SLURM_NNODES:-}}"
if [[ -z "$nodes" ]]; then
    job="$(scontrol show job "$SLURM_JOB_ID" --oneliner)"
    [[ "$job" =~ NumNodes=([0-9]+) ]] || { echo "Cannot determine allocation node count" >&2; exit 2; }
    nodes="${BASH_REMATCH[1]}"
fi
if [[ ! "$nodes" =~ ^[1-9][0-9]*$ ]] || (( nodes < 2 )); then
    echo "This demo requires two allocated nodes. From a login node, run SC26_NODES=2 bash frontier/allocate.sh." >&2
    exit 2
fi
export SC26_NODES=2 SC26_GPUS_PER_TASK="$gpus" JULIA_PKG_OFFLINE=true
exec bash "$root/frontier/launch.sh" 2 "$root/demos/$demo" "$@"
