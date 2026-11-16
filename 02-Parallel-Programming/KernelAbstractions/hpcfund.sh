#!/usr/bin/env bash
# One-command setup for the GPU notebooks (automates the setup steps in README.md).
#
#   ./hpcfund.sh local   [folder]               # your own Mac / GPU machine
#   ./hpcfund.sh cluster [username] [folder]    # AMD HPC Fund cluster
#
# folder: notebook folder inside the repo (default: KernelAbstractions).
#         A bare name is looked up under 02-Parallel-Programming/,
#         e.g. "KernelTuner/proposed_solution".
#
# Cluster mode, run from your laptop:
#   1. opens one SSH connection (password/2FA asked once, reused afterwards)
#   2. installs Julia, clones/updates the repo, installs the "GPU Julia" kernel
#   3. submits job.notebook and waits for Jupyter to start on the compute node
#   4. tunnels the Jupyter port to your laptop and opens your browser
#   5. on Ctrl-C (or closing the terminal) cancels the job: no forgotten nodes
#
# Options (environment variables):
#   HPCFUND_HOST=hpcfund.amd.com   login node
#   HPCFUND_SKIP_SETUP=1           skip step 2 (faster reconnects)
#   HPCFUND_KEEP_JOB=1             do not cancel the job on exit
set -euo pipefail

REPO_URL="https://github.com/JuliaParallel/julia-hpc-tutorial-sc26.git"
REPO_NAME="julia-hpc-tutorial-sc26"
HOST="${HPCFUND_HOST:-hpcfund.amd.com}"
JOB_TEMPLATE="/work1/proj_sc26julia/shared/job.notebook"
KERNEL_NAME="GPU Julia"

say()  { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33mwarning:\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31merror:\033[0m %s\n' "$*" >&2; exit 1; }

usage() { awk 'NR>1 && /^#/ {sub(/^# ?/, ""); print; next} NR>1 {exit}' "$0"; exit "${1:-0}"; }

# "KernelTuner" -> "02-Parallel-Programming/KernelTuner"
folder_path() {
    local f="${1:-KernelAbstractions}"
    case "$f" in
        02-*|*/02-*) printf '%s\n' "$f" ;;
        *)           printf '02-Parallel-Programming/%s\n' "$f" ;;
    esac
}

open_url() {
    local url="$1"
    if command -v open >/dev/null 2>&1; then open "$url"
    elif command -v wslview >/dev/null 2>&1; then wslview "$url"
    elif command -v xdg-open >/dev/null 2>&1; then xdg-open "$url" >/dev/null 2>&1
    else return 1
    fi
}

port_free() { ! (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null; }

# ---------------------------------------------------------------- local mode
run_local() {
    local rel; rel="$(folder_path "${1:-}")"
    local script_dir; script_dir="$(cd "$(dirname "$0")" && pwd)"
    local root

    # Use the checkout this script lives in, otherwise clone next to it.
    if root="$(git -C "$script_dir" rev-parse --show-toplevel 2>/dev/null)" \
        && [[ -d "$root/02-Parallel-Programming" ]]; then
        say "Using repository at $root"
    else
        root="$PWD/$REPO_NAME"
        if [[ -d "$root/.git" ]]; then
            say "Updating $root"; git -C "$root" pull --ff-only || warn "git pull failed; using existing files"
        else
            say "Cloning repository into $root"; git clone "$REPO_URL" "$root"
        fi
    fi

    if ! command -v julia >/dev/null 2>&1; then
        export PATH="$HOME/.juliaup/bin:$PATH"
        if ! command -v julia >/dev/null 2>&1; then
            say "Installing Julia (juliaup)"
            curl -fsSL https://install.julialang.org | sh -s -- --yes
            export PATH="$HOME/.juliaup/bin:$PATH"
        fi
    fi

    [[ -f "$root/$rel/open_notebook.jl" ]] || die "no open_notebook.jl in $root/$rel"
    cd "$root/$rel"
    say "Starting Jupyter in $rel (pick kernel \"$KERNEL_NAME\", set DEV in the second cell)"
    exec julia --project=. open_notebook.jl
}

# -------------------------------------------------------------- cluster mode
JOB_ID=""
SSH_CTL=""
TARGET=""

rsh() { ssh -o ControlPath="$SSH_CTL" "$TARGET" "$@"; }

cleanup() {
    local status=$?
    trap - EXIT INT TERM HUP
    if [[ -n "$JOB_ID" && "${HPCFUND_KEEP_JOB:-0}" != 1 ]]; then
        say "Cancelling job $JOB_ID (releasing the node)"
        rsh "scancel $JOB_ID" || warn "could not cancel job $JOB_ID; run 'scancel $JOB_ID' on the cluster"
    elif [[ -n "$JOB_ID" ]]; then
        say "Leaving job $JOB_ID running. Cancel it later with: ssh $TARGET scancel $JOB_ID"
    fi
    if [[ -n "$SSH_CTL" ]]; then
        ssh -o ControlPath="$SSH_CTL" -O exit "$TARGET" >/dev/null 2>&1 || true
    fi
    exit "$status"
}

# Print "STATE NODELIST" for the job; empty output = job left the queue.
# Retries ssh hiccups, so a dropped packet is never mistaken for a finished job.
job_state() {
    local tries=0 res
    while :; do
        if res="$(rsh "squeue -h -j $JOB_ID -o '%T %N'" 2>/dev/null)"; then
            printf '%s\n' "$res"; return 0
        fi
        tries=$((tries + 1))
        (( tries < 6 )) || die "lost contact with $TARGET while watching job $JOB_ID"
        sleep 5
    done
}

run_cluster() {
    local user="${1:-}" rel
    rel="$(folder_path "${2:-}")"
    TARGET="${user:+$user@}$HOST"

    trap cleanup EXIT
    trap 'exit 130' INT TERM HUP

    # 1. One authenticated connection, reused by every later ssh call.
    SSH_CTL="$HOME/.ssh/hpcfund-%C"
    say "Connecting to $TARGET (you may be asked for your password once)"
    ssh -o ControlMaster=yes -o ControlPath="$SSH_CTL" -o ControlPersist=no \
        -o ServerAliveInterval=60 -fN "$TARGET" || die "ssh to $TARGET failed"

    # 2. Remote setup (idempotent).
    if [[ "${HPCFUND_SKIP_SETUP:-0}" != 1 ]]; then
        say "Setting up Julia, the repository and the \"$KERNEL_NAME\" kernel on the cluster"
        # ssh joins arguments into one remote command line, so quote them.
        local args; args="$(printf '%q ' "$REPO_URL" "$REPO_NAME" "$rel" "$JOB_TEMPLATE" "$KERNEL_NAME")"
        rsh "bash -s -- $args" <<'REMOTE'
set -euo pipefail
repo_url="$1" repo="$2" rel="$3" template="$4" kernel="$5"
export PATH="$HOME/.juliaup/bin:$PATH"
if ! command -v julia >/dev/null 2>&1; then
    echo "  installing Julia"
    curl -fsSL https://install.julialang.org | sh -s -- --yes
fi
cd "$HOME"
if [[ -d "$repo/.git" ]]; then
    echo "  updating $repo"
    git -C "$repo" pull --ff-only || echo "  (git pull failed; using existing files)"
else
    git clone "$repo_url" "$repo"
fi
cd "$repo/$rel"
# Prefer the job script shipped in the repo (as open_notebook.jl does).
if [[ -f "$HOME/$repo/amdhpcfund.job.notebook" ]]; then
    cp "$HOME/$repo/amdhpcfund.job.notebook" job.notebook
else
    cp "$template" job.notebook
fi
# IJulia goes into the default (global) environment, which stays on the load
# path under --project=., so the repo's Project/Manifest are never modified.
julia -e 'using Pkg; Base.find_package("IJulia") === nothing && Pkg.add("IJulia")'
julia --project=. -e "using Pkg; Pkg.instantiate(); using IJulia; IJulia.installkernel(\"$kernel\", \"--project=@.\", \"--threads=auto\")"
REMOTE
    fi

    # 3. Submit the notebook job.
    local remote_dir="\$HOME/$REPO_NAME/$rel"
    JOB_ID="$(rsh "cd $remote_dir && sbatch --parsable job.notebook" | tail -n1 | cut -d';' -f1)"
    [[ "$JOB_ID" =~ ^[0-9]+$ ]] || { local bad="$JOB_ID"; JOB_ID=""; die "sbatch did not return a job id: $bad"; }
    say "Submitted job $JOB_ID; waiting for a node (Ctrl-C cancels it)"

    local out
    out="$(rsh "scontrol show job $JOB_ID" | sed -n 's/.*StdOut=\([^ ]*\).*/\1/p' | head -n1)"
    [[ -n "$out" ]] || out="$remote_dir/job.$JOB_ID.out"

    # 4. Wait until the job runs and Jupyter prints its URL.
    local state="" node="" last="" url=""
    while :; do
        local st; st="$(job_state)" || exit 1
        read -r state node <<<"$st"
        if [[ -z "$state" ]]; then
            warn "job $JOB_ID is no longer in the queue. Last lines of its output:"
            rsh "tail -n 30 $out" >&2 || true
            exit 1
        fi
        [[ "$state" != "$last" ]] && say "Job $JOB_ID: $state${node:+ on $node}" && last="$state"
        if [[ "$state" == RUNNING ]]; then
            url="$(rsh "grep -Eo 'https?://[^ ]+[?&]token=[A-Za-z0-9]+' $out 2>/dev/null | tail -n1" || true)"
            [[ -n "$url" ]] && break
        fi
        sleep 10
    done
    node="${node%%,*}"   # first node if Slurm lists several

    # 5. Tunnel to Jupyter through the existing connection.
    # Jupyter listens only on the node's 127.0.0.1, so it cannot be reached
    # as node:PORT. job.notebook instead opens a reverse tunnel from the node
    # to login1 and prints "ssh ... -L <local>:localhost:<login_port>";
    # forward to that login_port on the login node.
    local hostport rport path lport login_port dest
    hostport="${url#*://}"; hostport="${hostport%%/*}"
    rport="${hostport##*:}"
    path="/${url#*://*/}"
    [[ "$rport" =~ ^[0-9]+$ ]] || rport=8888
    login_port="$(rsh "grep -Eo -- '-L [0-9]+:localhost:[0-9]+' $out 2>/dev/null | tail -n1" | sed 's/.*://' || true)"
    if [[ "$login_port" =~ ^[0-9]+$ ]]; then
        dest="localhost:$login_port"
        local login_host; login_host="$(rsh hostname)"
        [[ "$login_host" == login1* ]] || warn "connected to $login_host, but the job tunnels to login1; if the page does not load, set HPCFUND_HOST to login1's address"
    else
        dest="$node:$rport"   # job script without a reverse tunnel
    fi
    lport="$rport"
    while ! port_free "$lport"; do lport=$((lport + 1)); done
    ssh -o ControlPath="$SSH_CTL" -O forward -L "$lport:$dest" "$TARGET" >/dev/null || die "could not open the SSH tunnel"
    local local_url="http://localhost:$lport$path"

    say "Jupyter is running on $node"
    echo
    echo "    $local_url"
    echo
    echo "    In Jupyter pick kernel \"$KERNEL_NAME\" and set DEV = :AMDGPU in the second cell."
    echo "    Press Ctrl-C here when you are done; the job will be cancelled."
    echo
    open_url "$local_url" || true

    # Stay alive until Ctrl-C or until the job ends (e.g. time limit).
    local st2
    while st2="$(job_state)" && [[ -n "$st2" ]]; do sleep 30; done
    [[ -z "${st2:-}" ]] || exit 1
    warn "job $JOB_ID has ended. Last lines of its output:"
    rsh "tail -n 20 $out" >&2 || true
}

# ---------------------------------------------------------------------- main
case "${1:-}" in
    local)          shift; run_local "$@" ;;
    cluster)        shift; run_cluster "$@" ;;
    -h|--help|"")   usage 0 ;;
    *)              usage 1 ;;
esac
