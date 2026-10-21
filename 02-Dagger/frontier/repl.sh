#!/bin/bash
# Keep this process alive between edits to retain compiled code.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
export SC26_PTY=1
exec bash "$root/frontier/launch.sh" 1 -i "$root/harness/session.jl"
