#!/bin/bash
# Keep this process alive between edits to retain compiled code.
# Run from the login-node allocation shell so preparation can download packages.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
: "${SLURM_JOB_ID:?Obtain an allocation with frontier/allocate.sh first}"
bash "$root/frontier/prepare.sh"
export SC26_PTY=1
exec bash "$root/frontier/launch.sh" 1 -i "$root/harness/session.jl"
