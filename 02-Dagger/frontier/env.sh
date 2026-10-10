# Environment for the Dagger segment on Odo (Frontier's training system).
#   source frontier/env.sh
#
# Anything marked TODO(confirm) is a guess until checked with OLCF / the
# tutorial organizers.

# --- Modules: TODO(confirm) exact versions available on Odo in Nov 2026 ---
module load PrgEnv-gnu
module load rocm                      # TODO(confirm): pin e.g. rocm/6.x validated with AMDGPU.jl
module load cray-mpich
module load craype-accel-amd-gfx90a   # MI250X

# --- Julia: TODO(confirm) OLCF module vs. a shared juliaup install ---
# module load julia
export PATH="${SC26_JULIA_BIN:-/lustre/orion/TODO/world-shared/sc26/julia/bin}:$PATH"

# --- Package depot ---
# Your own depot first (writable), then the shared, pre-compiled depot the
# instructors built with frontier/setup.sh. This is what makes the first
# `using Dagger` take seconds instead of minutes.
export SC26_SHARED_DEPOT="${SC26_SHARED_DEPOT:-/lustre/orion/TODO/world-shared/sc26/depot}"
export JULIA_DEPOT_PATH="$HOME/.julia-sc26:$SC26_SHARED_DEPOT"

export SC26_ROOT="${SC26_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export JULIA_PROJECT="$SC26_ROOT"

# --- GPU + MPI ---
export ROCM_PATH="${ROCM_PATH:-${OLCF_ROCM_ROOT:-/opt/rocm}}"
# Send GPU data over MPI through host memory (works with any MPI build). GPU-aware
# MPI would also need MPICH_GPU_SUPPORT_ENABLED=1 plus the Cray GTL library.
export DAGGER_MPI_GPU_DIRECT=0
