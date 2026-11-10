# Source from Bash on andromeda (one Slurm node, tau01: 48 cores, two NVIDIA
# A30 GPUs, partition gpu). Override SC26_* before sourcing when needed.
export SC26_ROOT="${SC26_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export SC26_CACHE_ROOT="${SC26_CACHE_ROOT:-$HOME/sc26-dagger}"
# Odo's course module is julia/1.13.1; andromeda has no Julia, so prepare.sh
# downloads this official build into the cache.
export SC26_JULIA_VERSION="${SC26_JULIA_VERSION:-1.13.1}"
export SC26_JULIA_HOME="${SC26_JULIA_HOME:-$SC26_CACHE_ROOT/julia-$SC26_JULIA_VERSION}"
export SC26_DEPOT="${SC26_DEPOT:-$SC26_CACHE_ROOT/depot}"
export SC26_PARTITION="${SC26_PARTITION:-gpu}"
export PATH="$SC26_JULIA_HOME/bin:$PATH"
# The trailing ':' retains Julia's bundled depots.
export JULIA_DEPOT_PATH="$SC26_DEPOT:"
export JULIA_PROJECT="$SC26_ROOT"
export JULIA_NUM_PRECOMPILE_TASKS="${JULIA_NUM_PRECOMPILE_TASKS:-8}"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
# Host-staged GPU MPI transfers, as on Odo.
export DAGGER_MPI_GPU_DIRECT="${DAGGER_MPI_GPU_DIRECT:-0}"
export DAGGER_IPC="${DAGGER_IPC:-false}"
export JULIA_DAGGER_MPI_WARN_PERIOD="${JULIA_DAGGER_MPI_WARN_PERIOD:-60}"
