<#
.SYNOPSIS
  One-command setup for the GPU notebooks on Windows (PowerShell version of hpcfund.sh).

.DESCRIPTION
  Run from PowerShell (Windows 10/11, uses the built-in ssh):

    powershell -ExecutionPolicy Bypass -File hpcfund.ps1 local   [-Folder <folder>]
    powershell -ExecutionPolicy Bypass -File hpcfund.ps1 cluster <username> [-Folder <folder>]

  cluster mode:
    1. logs in (password #1), installs Julia, clones/updates the repo,
       installs the "GPU Julia" kernel, submits the notebook job and waits
       until Jupyter runs on a compute node
    2. opens the tunnel (password #2) and your browser
    3. Ctrl-C cancels the job and closes the tunnel

  Folder: notebook folder inside the repo (default KernelAbstractions); a bare
  name is looked up under 02-Parallel-Programming\, e.g. KernelTuner/proposed_solution.

  -SkipSetup   skip installing/updating (faster reconnects)
  -KeepJob     do not cancel the job on Ctrl-C
  -HostName    login node (default hpcfund.amd.com)
#>
param(
    [Parameter(Position = 0)][ValidateSet('local', 'cluster')][string]$Mode,
    [Parameter(Position = 1)][string]$Username,
    [string]$Folder = 'KernelAbstractions',
    [string]$HostName = 'hpcfund.amd.com',
    [switch]$SkipSetup,
    [switch]$KeepJob
)

$ErrorActionPreference = 'Stop'
$RepoUrl    = 'https://github.com/JuliaParallel/julia-hpc-tutorial-sc26.git'
$RepoName   = 'julia-hpc-tutorial-sc26'
$KernelName = 'GPU Julia'

function Say($msg)  { Write-Host "==> $msg" -ForegroundColor Cyan }
function Fail($msg) { Write-Host "error: $msg" -ForegroundColor Red; exit 1 }

if (-not $Mode) { Get-Help $PSCommandPath -Detailed; exit 0 }

$Rel = if ($Folder -like '02-*') { $Folder } else { "02-Parallel-Programming/$Folder" }

function Test-PortFree([int]$port) {
    try {
        $l = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, $port)
        $l.Start(); $l.Stop(); return $true
    } catch { return $false }
}

# ---------------------------------------------------------------- local mode
if ($Mode -eq 'local') {
    $root = $null
    $here = Split-Path -Parent $PSCommandPath
    if (Get-Command git -ErrorAction SilentlyContinue) {
        $top = & git -C $here rev-parse --show-toplevel 2>$null
        if ($LASTEXITCODE -eq 0 -and (Test-Path (Join-Path $top '02-Parallel-Programming'))) { $root = $top }
    }
    if (-not $root) {
        $root = Join-Path (Get-Location) $RepoName
        if (Test-Path (Join-Path $root '.git')) {
            Say "Updating $root"; & git -C $root pull --ff-only
        } elseif (Get-Command git -ErrorAction SilentlyContinue) {
            Say "Cloning repository into $root"; & git clone $RepoUrl $root
        } else {
            Say "git not found: downloading the repository as a zip"
            $zip = Join-Path $env:TEMP "$RepoName.zip"
            Invoke-WebRequest "https://github.com/JuliaParallel/$RepoName/archive/refs/heads/main.zip" -OutFile $zip
            Expand-Archive $zip -DestinationPath (Get-Location) -Force
            Rename-Item (Join-Path (Get-Location) "$RepoName-main") $RepoName
        }
    }
    if (-not (Get-Command julia -ErrorAction SilentlyContinue)) {
        Fail "Julia is not installed. Install it with:  winget install --id 9NJNWW8PVKMN -e -s msstore  (or from https://julialang.org/downloads/), then open a new PowerShell and run this again."
    }
    $dir = Join-Path $root $Rel
    if (-not (Test-Path (Join-Path $dir 'open_notebook.jl'))) { Fail "no open_notebook.jl in $dir" }
    Set-Location $dir
    Say "Starting Jupyter in $Rel (pick kernel `"$KernelName`", set DEV in the second cell)"
    & julia --project=. open_notebook.jl
    exit $LASTEXITCODE
}

# -------------------------------------------------------------- cluster mode
if (-not (Get-Command ssh -ErrorAction SilentlyContinue)) {
    Fail "ssh not found. Enable it in Settings > System > Optional features > OpenSSH Client."
}
$Target = if ($Username) { "$Username@$HostName" } else { $HostName }

# Runs on the login node: setup, submit, wait for Jupyter, print the result.
# Ctrl-C arrives through the terminal (ssh -tt) and cancels the job.
$Remote = @'
set -euo pipefail
repo_url="$1" repo="$2" rel="$3" kernel="$4" skip="$5"
template=/work1/proj_sc26julia/shared/job.notebook
export PATH="$HOME/.juliaup/bin:$PATH"
cd "$HOME"
if [[ "$skip" != 1 ]]; then
    echo "==> Setting up Julia, the repository and the \"$kernel\" kernel on the cluster"
    if ! command -v julia >/dev/null 2>&1; then
        curl -fsSL https://install.julialang.org | sh -s -- --yes
    fi
    if [[ -d "$repo/.git" ]]; then
        git -C "$repo" pull --ff-only || echo "  (git pull failed; using existing files)"
    else
        git clone "$repo_url" "$repo"
    fi
fi
cd "$HOME/$repo/$rel"
if [[ "$skip" != 1 || ! -f job.notebook ]]; then
    if [[ -f "$HOME/$repo/amdhpcfund.job.notebook" ]]; then
        cp "$HOME/$repo/amdhpcfund.job.notebook" job.notebook
    else
        cp "$template" job.notebook
    fi
fi
if [[ "$skip" != 1 ]]; then
    julia -e 'using Pkg; Base.find_package("IJulia") === nothing && Pkg.add("IJulia")'
    julia --project=. -e "using Pkg; Pkg.instantiate(); using IJulia; IJulia.installkernel(\"$kernel\", \"--project=@.\", \"--threads=auto\")"
fi
job="$(sbatch --parsable job.notebook | tail -n1 | cut -d';' -f1)"
[[ "$job" =~ ^[0-9]+$ ]] || { echo "error: sbatch did not return a job id: $job"; exit 1; }
trap 'echo; echo "==> Cancelling job $job (releasing the node)"; scancel $job; exit 130' INT HUP TERM
echo "==> Submitted job $job; waiting for a node (Ctrl-C cancels it)"
out="$(scontrol show job "$job" | sed -n 's/.*StdOut=\([^ ]*\).*/\1/p' | head -n1)"
[[ -n "$out" ]] || out="job.$job.out"
last="" url=""
while :; do
    read -r state node <<<"$(squeue -h -j "$job" -o '%T %N' 2>/dev/null || true)"
    if [[ -z "$state" ]]; then
        echo "error: job $job is no longer in the queue. Last lines of its output:"
        tail -n 30 "$out" 2>/dev/null || true
        exit 1
    fi
    if [[ "$state" != "$last" ]]; then echo "==> Job $job: $state${node:+ on $node}"; last="$state"; fi
    if [[ "$state" == RUNNING ]]; then
        url="$(grep -Eo 'https?://[^ ]+[?&]token=[A-Za-z0-9]+' "$out" 2>/dev/null | tail -n1 || true)"
        [[ -n "$url" ]] && break
    fi
    sleep 10
done
login_port="$(grep -Eo -- '-L [0-9]+:localhost:[0-9]+' "$out" | tail -n1 | sed 's/.*://' || true)"
hostport="${url#*://}"; hostport="${hostport%%/*}"
echo "HPCFUND_RESULT $job ${login_port:--} ${node%%,*} ${hostport##*:} /${url#*://*/} $(hostname)"
'@

# Send the script base64-encoded: avoids Windows CRLF and quoting problems.
$Bytes = [System.Text.Encoding]::UTF8.GetBytes(($Remote -replace "`r", ''))
$B64   = [Convert]::ToBase64String($Bytes)
$Skip  = if ($SkipSetup) { '1' } else { '0' }
$Cmd   = "echo $B64 | base64 -d > /tmp/hpcfund.`$`$.sh && bash /tmp/hpcfund.`$`$.sh '$RepoUrl' '$RepoName' '$Rel' '$KernelName' $Skip; s=`$?; rm -f /tmp/hpcfund.`$`$.sh; exit `$s"

Say "Connecting to $Target (password 1 of 2)"
$result = $null
& ssh -tt $Target $Cmd | ForEach-Object {
    $line = $_.TrimEnd("`r")
    if ($line -like 'HPCFUND_RESULT *') { $result = $line } else { Write-Host $line }
}
if (-not $result) { Fail "the job did not start Jupyter (see the messages above)" }

# HPCFUND_RESULT <job> <login_port|-> <node> <jupyter_port> <path> <login_host>
$f = $result -split ' '
$Job = $f[1]; $LoginPort = $f[2]; $Node = $f[3]; $JPort = $f[4]; $UrlPath = $f[5]; $LoginHost = $f[6]

# Jupyter only listens on the node's 127.0.0.1; job.notebook tunnels it to
# login1:<login_port>, so forward to that port on the login node.
if ($LoginPort -match '^\d+$') {
    $Dest = "localhost:$LoginPort"; $LocalPort = [int]$LoginPort
    if ($LoginHost -notlike 'login1*') { Write-Host "warning: connected to $LoginHost but the job tunnels to login1; if the page does not load, use -HostName with login1's address" -ForegroundColor Yellow }
} else {
    $Dest = "${Node}:$JPort"; $LocalPort = [int]$JPort
}
while (-not (Test-PortFree $LocalPort)) { $LocalPort++ }
$Url = "http://localhost:$LocalPort$UrlPath"

Write-Host ""
Say "Jupyter is running on $Node"
Write-Host ""
Write-Host "    $Url"
Write-Host ""
Write-Host "    In Jupyter pick kernel `"$KernelName`" and set DEV = :AMDGPU in the second cell."
Write-Host "    Press Ctrl-C here when you are done; the job will be cancelled."
Write-Host "    (If the page does not load right away, refresh it after entering the password.)"
Write-Host ""

# Open the browser a few seconds from now, once the tunnel is up.
Start-Job -ScriptBlock { param($u) Start-Sleep 8; Start-Process $u } -ArgumentList $Url | Out-Null

# Tunnel; the remote side waits for the job and cancels it on Ctrl-C or disconnect.
if ($KeepJob) {
    $Hold = "echo Tunnel open. Ctrl-C closes it, job $Job keeps running.; while squeue -h -j $Job -o %T | grep -q .; do sleep 30; done"
} else {
    $Hold = "trap 'echo; echo Cancelling job $Job and releasing the node; scancel $Job; exit' INT HUP TERM; while squeue -h -j $Job -o %T | grep -q .; do sleep 30; done; echo Job $Job has ended."
}
Say "Opening the tunnel (password 2 of 2)"
& ssh -tt -o ServerAliveInterval=60 -o ExitOnForwardFailure=yes -L "${LocalPort}:$Dest" $Target $Hold
