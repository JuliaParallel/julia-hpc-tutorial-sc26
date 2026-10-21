#!/bin/bash
# Persistent MPI harness prompt (options, not arbitrary Julia expressions).
# SC26_NODES=1 bash frontier/mpi-repl.sh 4
# allocate.sh prepares packages on the login node before obtaining resources.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
np="${1:-2}"
: "${SLURM_JOB_ID:?Obtain an allocation with frontier/allocate.sh first}"
export SC26_PTY=1
export JULIA_PKG_OFFLINE=true
exec bash "$root/frontier/launch.sh" "$np" "$root/harness/mpi_session.jl"
