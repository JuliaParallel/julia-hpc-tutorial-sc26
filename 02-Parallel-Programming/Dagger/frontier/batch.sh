#!/bin/bash
# sbatch -A trn047 -N1 --gpus-per-node=8 -t 00:30:00 frontier/batch.sh 4 --mpi --target=gpu --impl=solution
#SBATCH --job-name=sc26-dagger
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --threads-per-core=1
#SBATCH --gpus-per-node=8
#SBATCH --time=00:30:00
#SBATCH --output=sc26-dagger-%j.log
set -euo pipefail
export SC26_NODES="${SC26_NODES:-$SLURM_JOB_NUM_NODES}"
root="${SC26_ROOT:-$SLURM_SUBMIT_DIR}"
exec bash "$root/frontier/run.sh" "$@"
