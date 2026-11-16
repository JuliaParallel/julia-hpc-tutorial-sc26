#!/usr/bin/env bash
export PATH="$HOME/.juliaup/bin:$PATH"
export SC26_ROOT="${SC26_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export SC26_CACHE_ROOT="${SC26_CACHE_ROOT:-$HOME/sc26-dagger}"
export JULIA_DEPOT_PATH="$SC26_CACHE_ROOT/cpu-depot:"
export JULIA_PROJECT="$SC26_ROOT/cpu"
export JULIA_NUM_THREADS="${SC26_THREADS:-$(nproc)}"
export JULIA_NUM_PRECOMPILE_TASKS="${JULIA_NUM_PRECOMPILE_TASKS:-2}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
