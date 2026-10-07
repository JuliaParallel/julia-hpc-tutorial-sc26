#!/bin/bash
# Run the harness on your allocation's compute node.
#   bash frontier/run.sh 1 --impl=student                 # 1 process, CPU threads
#   bash frontier/run.sh 1 --impl=student --target=gpu    # 1 process, the GPU
#   bash frontier/run.sh 2 --impl=student --mpi --strict  # 2 MPI ranks
# The first argument is the number of processes (MPI ranks).
set -euo pipefail
source "$(dirname "$0")/env.sh"
NP="$1"; shift
CPUS="${SLURM_CPUS_PER_TASK:-3}"
if [ "$NP" = 1 ]; then CPUS=$(( CPUS * ${SLURM_NTASKS:-1} )); fi   # one rank gets all cores
# TODO(confirm on Odo): both ranks sharing the job's single GPU via --gpu-bind=none
exec srun -n "$NP" -c "$CPUS" --gpus=1 --gpu-bind=none \
     julia --project="$SC26_ROOT" -t "$CPUS" "$SC26_ROOT/harness/run.jl" "$@"
