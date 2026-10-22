#!/bin/bash
# Launch Julia in an existing allocation, including from its login-node shell:
# SLURM_JOB_ID=123 SC26_NODES=1 bash frontier/launch.sh 4 demos/04_mpi.jl
set -euo pipefail
source "$(dirname "$0")/env.sh"
: "${SLURM_JOB_ID:?Obtain an allocation with frontier/allocate.sh first}"
np="${1:?Usage: launch.sh RANKS [Julia arguments]}"; shift
nodes="${SC26_NODES:-1}"
threads="${SC26_THREADS:-7}"
[[ "$np" =~ ^[1-9][0-9]*$ && "$nodes" =~ ^[1-9][0-9]*$ ]] || { echo "Ranks/nodes must be positive integers" >&2; exit 2; }
(( np % nodes == 0 )) || { echo "Ranks must divide evenly across nodes" >&2; exit 2; }
rpn=$((np / nodes))
gpt="${SC26_GPUS_PER_TASK:-1}"
bind=closest
if (( np == 1 )); then gpt="${SC26_GPUS_PER_TASK:-8}"; bind=none; fi
# "closest" can expose only one GPU even when a rank requested several.
if (( np > 1 && gpt > 1 )); then bind="per_task:$gpt"; fi
if (( gpt == 0 )); then bind=none; fi
(( rpn * gpt <= 8 )) || { echo "Requested more than eight GPUs per node" >&2; exit 2; }
cd "$SC26_ROOT"
terminal=()
if [[ "${SC26_PTY:-0}" == 1 ]]; then terminal+=(--pty); fi
limit=()
if [[ -n "${SC26_STEP_TIME:-}" ]]; then limit+=(--time="$SC26_STEP_TIME"); fi
exec srun --jobid="$SLURM_JOB_ID" --exact -N "$nodes" -n "$np" \
    --ntasks-per-node="$rpn" -c "$threads" --threads-per-core=1 \
    --gpus-per-task="$gpt" --gpu-bind="$bind" --cpu-bind=cores \
    --kill-on-bad-exit=1 "${terminal[@]}" "${limit[@]}" julia --project="$SC26_ROOT" -t "$threads" "$@"
