#!/bin/bash
# Harness wrapper: run.sh RANKS [harness options].
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
np="${1:?Usage: run.sh RANKS [harness options]}"; shift
exec bash "$root/frontier/launch.sh" "$np" "$root/harness/run.jl" "$@"
