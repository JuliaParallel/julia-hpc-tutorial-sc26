#!/bin/bash
# Run inside a compute step, without nesting another srun:
# srun -N1 -n1 -c7 --gpus-per-task=8 --gpu-bind=none bash frontier/setup.sh
# Set SC26_DEPOT to the course depot to prepare it for attendees.
set -euo pipefail
source "$(dirname "$0")/env.sh"
mkdir -p "$SC26_DEPOT"
# Download packages and write MPI preferences with prepare.sh on a login node first.
export JULIA_PKG_OFFLINE=true
julia --project="$SC26_ROOT" -e '
    using Pkg
    Pkg.precompile()
    using Dagger, AMDGPU, MPI, UnicodePlots
    println("Dagger ", pkgversion(Dagger))
    AMDGPU.functional() || error("AMDGPU is not functional")
    println("Visible AMD GPUs: ", AMDGPU.devices())
    println("MPI: ", MPI.Get_library_version())
'
