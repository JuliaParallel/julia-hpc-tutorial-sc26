#!/bin/bash
# Course layout: one whole node per independent attendee or presenter.
set -euo pipefail
source "$(dirname "$0")/env.sh"
nodes="${SC26_NODES:-1}"
args=(-A "$SC26_ACCOUNT" -p batch -N "$nodes" -n "$((nodes * 8))"
      --ntasks-per-node=8 -c 7 --threads-per-core=1 --gpus-per-node=8
      -t "${SC26_TIME:-02:00:00}" -J sc26-dagger)
if [[ -n "${SC26_RESERVATION:-}" ]]; then args+=(--reservation="$SC26_RESERVATION"); fi
exec salloc "${args[@]}" "$@"
