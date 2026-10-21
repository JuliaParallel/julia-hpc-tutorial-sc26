#!/bin/bash
# Run on the laptop, after an Odo benchmark. Export charts locally.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
remote="${SC26_REMOTE_ROOT:-/gpfs/wolf2/olcf/${SC26_ACCOUNT:-trn047}/scratch/$USER/sc26-dagger/02-Dagger}"
mkdir -p "$root/results"
# Odo permits one session per transport. Keep transfers off the interactive
# terminal's transport, while still authenticating only once per idle lifetime.
socket="${SC26_FETCH_SOCKET:-$HOME/.ssh/control/odo-results}"
if [[ ! -d "$(dirname "$socket")" ]]; then mkdir -m 700 -p "$(dirname "$socket")"; fi
if ! ssh -S "$socket" -O check odo >/dev/null 2>&1; then
    if command -v odo-connect >/dev/null 2>&1; then
        odo-connect "$socket"
    else
        ssh -S "$socket" -o ControlMaster=auto -o ControlPersist=8h -MNf odo
    fi
fi
scp -o ControlPath="$socket" "odo:$remote/results/results.csv" "$root/results/odo-results.csv"
python3 "$root/harness/plot.py" "$root/results/odo-results.csv" \
    --output "$root/results/odo-performance" "$@"
