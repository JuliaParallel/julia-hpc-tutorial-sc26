#!/bin/bash
# Compile the packages in a Slurm job with one GPU, after prepare.sh, and
# check that CUDA and MPI work. Later runs start from the compiled cache.
#   bash andromeda/precompile.sh
set -euo pipefail
source "$(dirname "$0")/env.sh"
export JULIA_PKG_OFFLINE=true
cd "$SC26_ROOT"
args=(-p "$SC26_PARTITION" -N 1 -n 1 -c 16 --mem=64G --gpus-per-task=1
      -t "${SC26_PRECOMPILE_TIME:-01:00:00}" -J sc26-dagger-precompile --input=none)
if [[ -n "${SC26_ACCOUNT:-}" ]]; then args+=(-A "$SC26_ACCOUNT"); fi
if [[ -n "${SC26_RESERVATION:-}" ]]; then args+=(--reservation="$SC26_RESERVATION"); fi
launch=(srun "${args[@]}")
# SC26_LAUNCHER=local compiles without Slurm, for testing only.
if [[ "${SC26_LAUNCHER:-slurm}" == local ]]; then launch=(); fi
exec "${launch[@]}" julia --project="$SC26_ROOT" --startup-file=no -e '
    using Pkg
    Pkg.precompile()
    using Dagger, CUDA, MPI, JSON3, UnicodePlots, Highlights
    println("Dagger ", pkgversion(Dagger), " · CUDA.jl ", pkgversion(CUDA))
    CUDA.functional() || error("CUDA is not functional in a GPU job")
    println("GPU: ", CUDA.name(CUDA.device()), " · driver ", CUDA.driver_version(),
            " · runtime ", CUDA.runtime_version())
    println("MPI: ", first(split(MPI.Get_library_version(), "\n")))
'
