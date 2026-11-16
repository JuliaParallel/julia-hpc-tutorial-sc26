#!/bin/bash
# Keep this process alive between edits to retain compiled code.
# allocate.sh prepares packages on the login node before obtaining resources.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
: "${SLURM_JOB_ID:?Obtain an allocation with frontier/allocate.sh first}"
export SC26_PTY=1
export JULIA_PKG_OFFLINE=true
exec bash "$root/frontier/launch.sh" 1 -i "$root/harness/session.jl"
