#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
mkdir -p "$SC26_CACHE_ROOT"
exec 9>"$SC26_CACHE_ROOT/cpu-prepare.lock"
flock 9
seed="$SC26_CACHE_ROOT/cpu-Manifest.toml"
if [[ ! -f "$JULIA_PROJECT/Manifest.toml" && -f "$seed" ]]; then
    cp "$seed" "$JULIA_PROJECT/Manifest.toml"
fi
export JULIA_PKG_PRECOMPILE_AUTO=0 JULIA_PKG_OFFLINE=false
julia --startup-file=no --project="$JULIA_PROJECT" -e '
    using Pkg
    isempty(Pkg.Registry.reachable_registries()) && Pkg.Registry.add("General")
    Pkg.resolve()
    Pkg.instantiate(; allow_autoprecomp=false)
    using MPIPreferences
    MPIPreferences.use_jll_binary("MPICH_jll")
    Pkg.precompile()
    using Dagger, MPI, Highlights, UnicodePlots
    println("Dagger ", pkgversion(Dagger), " · CPU fallback · ", Threads.nthreads(), " threads")
    println("MPI: ", MPI.Get_library_version())
'
cp "$JULIA_PROJECT/Manifest.toml" "$seed.tmp"
mv "$seed.tmp" "$seed"
