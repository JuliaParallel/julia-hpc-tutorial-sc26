#!/bin/bash
# Persistent MPI harness prompt (options, not arbitrary Julia expressions).
# SC26_NODES=1 bash frontier/mpi-repl.sh 4
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
np="${1:-2}"
export SC26_PTY=1
exec bash "$root/frontier/launch.sh" "$np" "$root/harness/mpi_session.jl"
