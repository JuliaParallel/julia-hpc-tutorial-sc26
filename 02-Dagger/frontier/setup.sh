#!/bin/bash
# INSTRUCTORS, once, before the tutorial: build the shared pre-compiled depot.
# Run on an Odo COMPUTE node (so AMDGPU finds a GPU while precompiling):
#   salloc ... ; srun -n1 bash frontier/setup.sh
set -euo pipefail
source "$(dirname "$0")/env.sh"
export JULIA_DEPOT_PATH="$SC26_SHARED_DEPOT"    # write into the shared depot only

julia --project="$SC26_ROOT" -e '
    using Pkg
    Pkg.instantiate()
    # Use Cray MPICH instead of the bundled MPICH (writes LocalPreferences.toml)
    using MPIPreferences
    MPIPreferences.use_system_binary(; library_names=["libmpi_cray"], mpiexec="srun")
    Pkg.precompile()
    using Dagger, AMDGPU, MPI, UnicodePlots
    println("Dagger ", pkgversion(Dagger), "  AMDGPU functional: ", AMDGPU.functional())
'
# Smoke test: every path the students will use
bash "$SC26_ROOT/frontier/run.sh" 1 --impl=solution --target=cpu --sizes=2048 --trials=1
bash "$SC26_ROOT/frontier/run.sh" 1 --impl=solution --target=gpu --sizes=8192 --trials=1
bash "$SC26_ROOT/frontier/run.sh" 2 --impl=solution --mpi --strict --sizes=2048 --trials=1
chmod -R a+rX "$SC26_SHARED_DEPOT"
