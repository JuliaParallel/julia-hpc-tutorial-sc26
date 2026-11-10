#!/bin/bash
# Submit one run to Slurm with the resources it needs, and wait for it. Each
# call is its own Slurm job; nothing stays allocated between runs.
#   bash andromeda/launch.sh 1 0 demos/01_tasks.jl               # CPU threads
#   bash andromeda/launch.sh 1 1 demos/03_gpu.jl                 # one GPU
#   bash andromeda/launch.sh 1 2 demos/05_multigpu.jl            # both GPUs
#   bash andromeda/launch.sh 2 0 demos/06_multinode.jl           # two MPI ranks
#   bash andromeda/launch.sh 2 2 demos/07_multinode_multigpu.jl  # one GPU per rank
#
# MPI ranks: srun starts one Slurm task per rank (--mpi=pmi2, which MPICH
# supports; pmix it does not), and Slurm binds each rank's CPUs and GPUs.
# SC26_MPI_LAUNCH=mpiexec instead makes one task hold every rank's CPUs and
# GPUs, with MPICH's own mpiexec starting the ranks inside it.
# SC26_LAUNCHER=local runs the command without Slurm, for testing only; its
# MPI ranks always start with mpiexec.
set -euo pipefail
source "$(dirname "$0")/env.sh"
usage="Usage: launch.sh RANKS GPUS [Julia arguments]"
np="${1:?$usage}"
gpus="${2:?$usage}"
shift 2
[[ "$np" =~ ^[1-9][0-9]*$ && "$gpus" =~ ^[0-9]+$ ]] || { echo "$usage" >&2; exit 2; }
(( gpus % np == 0 )) || { echo "Give every MPI rank the same number of GPUs" >&2; exit 2; }
threads="${SC26_THREADS:-$(( np == 1 ? 8 : 4 ))}"
launcher="${SC26_LAUNCHER:-slurm}"
if [[ "$launcher" == local ]]; then mpi_launch=mpiexec; else mpi_launch="${SC26_MPI_LAUNCH:-srun}"; fi
julia=(julia --project="$SC26_ROOT" -t "$threads" "$@")
export SC26_RANKS="$np" SC26_GPUS="$gpus"
cd "$SC26_ROOT"

if (( np > 1 )) && [[ "$mpi_launch" == mpiexec ]]; then
    command=(julia --project="$SC26_ROOT" --startup-file=no "$SC26_ROOT/andromeda/mpirun.jl" "$np"
             bash "$SC26_ROOT/andromeda/rank.sh" "${julia[@]}")
    tasks=1; cpus=$(( np * threads )); gpus_per_task="$gpus"
else
    command=("${julia[@]}")
    tasks="$np"; cpus="$threads"; gpus_per_task=$(( gpus / np ))
fi

if [[ "$launcher" == local ]]; then
    # Hide the GPUs a Slurm job would not get, as Slurm does.
    CUDA_VISIBLE_DEVICES="$(seq -s, 0 $(( gpus - 1 )) 2>/dev/null || true)"
    export CUDA_VISIBLE_DEVICES
    exec "${command[@]}"
fi

args=(-p "$SC26_PARTITION" -N 1 -n "$tasks" -c "$cpus" --threads-per-core=1
      --mem-per-cpu="${SC26_MEM_PER_CPU:-4G}" -t "${SC26_STEP_TIME:-00:20:00}"
      -J "${SC26_JOB_NAME:-sc26-dagger}" --cpu-bind=cores --kill-on-bad-exit=1)
if (( gpus_per_task > 0 )); then
    # Slurm sets CUDA_VISIBLE_DEVICES, so each task sees only its own GPUs.
    args+=(--gpus-per-task="$gpus_per_task" --gpu-bind=per_task:"$gpus_per_task")
fi
if (( tasks > 1 )); then args+=(--mpi="${SC26_SRUN_MPI:-pmi2}"); fi
# Only rank 0 reads commands; the other ranks receive them over MPI.
args+=(--input=0)
if [[ -n "${SC26_ACCOUNT:-}" ]]; then args+=(-A "$SC26_ACCOUNT"); fi
if [[ -n "${SC26_RESERVATION:-}" ]]; then args+=(--reservation="$SC26_RESERVATION"); fi
exec srun "${args[@]}" "${command[@]}"
