# Source from Bash on Odo. Override SC26_* before sourcing when needed.
module load PrgEnv-gnu "${SC26_JULIA_MODULE:-julia/1.13.1}" \
    "${SC26_ROCM_MODULE:-rocm/6.4.2}" cray-mpich craype-accel-amd-gfx90a

export SC26_ACCOUNT="${SC26_ACCOUNT:-trn047}"
export SC26_ROOT="${SC26_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export SC26_CACHE_ROOT="${SC26_CACHE_ROOT:-/gpfs/wolf2/olcf/$SC26_ACCOUNT/scratch/$USER/sc26-dagger}"
export SC26_DEPOT="${SC26_DEPOT:-$SC26_CACHE_ROOT/depot}"
# Instructors may point SC26_SHARED_DEPOT at the prepared course depot.
# The trailing ':' retains Julia's bundled depots.
export JULIA_DEPOT_PATH="$SC26_DEPOT:${SC26_SHARED_DEPOT:+$SC26_SHARED_DEPOT:}"
export JULIA_PROJECT="$SC26_ROOT"
export JULIA_NUM_PRECOMPILE_TASKS="${JULIA_NUM_PRECOMPILE_TASKS:-4}"
export ROCM_PATH="${ROCM_PATH:-${OLCF_ROCM_ROOT:-/opt/rocm-6.4.2}}"
export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH:-}:${ROCM_PATH}/lib:${LD_LIBRARY_PATH:-}"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
# Host-staged GPU MPI transfers; GPU-direct is an optional experiment.
export DAGGER_MPI_GPU_DIRECT="${DAGGER_MPI_GPU_DIRECT:-0}"
# Use host staging between local ranks too: ROCm IPC faulted with two GPUs/rank.
export DAGGER_IPC="${DAGGER_IPC:-false}"
export MPICH_GPU_SUPPORT_ENABLED="${MPICH_GPU_SUPPORT_ENABLED:-0}"
# Another rank's first JIT compilation exceeded the default 10-second warning
# interval in validation; retain the backend's timeout, but give compilation room.
export JULIA_DAGGER_MPI_WARN_PERIOD="${JULIA_DAGGER_MPI_WARN_PERIOD:-60}"
