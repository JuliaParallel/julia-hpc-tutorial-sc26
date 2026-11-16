# Shared-password workshop gateway

The deployed gateway runs the Odo relay and the AWS CPU allocator. Attendees
use one workshop password for both backends. They need Python 3.9+, OpenSSH,
a browser, and this repository on macOS, Linux, or Windows with WSL. On
Windows install Python and OpenSSH inside WSL. They need no AWS account,
AWS CLI, `awsmcp`, local Julia, SSH keys, or Odo
password. The local console obtains its gateway address and pinned SSH host
key from the public `host/workshop.json` descriptor.

## Attendee instructions

Use the latest merged code on **main**:

```bash
git clone --branch main https://github.com/JuliaParallel/julia-hpc-tutorial-sc26.git sc26-workshop
cd sc26-workshop/02-Parallel-Programming/Dagger
./console --gateway
```

On Windows, run `console.cmd --gateway` from Command Prompt, or
`.\console.cmd --gateway` from PowerShell. It starts the console in your default
WSL distribution. You can also run `./console --gateway` from a WSL terminal.
Both launchers forward all server arguments, for example
`./console --gateway --port 8770 --no-browser`. Relative paths are interpreted
in the current directory on macOS/Linux. The Windows launcher starts in the
Dagger folder; use WSL paths for file arguments (for example `/mnt/c/...`).

For an existing checkout, save or commit any tracked-file edits first, then run
`git switch main` and `git pull --ff-only origin main` before starting. If Git
is unavailable, [download the main ZIP](https://github.com/JuliaParallel/julia-hpc-tutorial-sc26/archive/refs/heads/main.zip),
extract it, and open its `02-Parallel-Programming/Dagger` folder. If your ZIP
extractor dropped executable permissions, run `chmod +x console` on macOS/Linux.

The console checks main at startup and every five minutes while the page is
open. A notice offers **Update and restart** when newer merged code exists;
**Check for updates** runs a check immediately. Disconnect and close any agent
terminals before updating. The server restarts on the same port with your
options, and the page reloads. Reconnect using the workshop password.
After restarting, the banner reports how many commits were pulled. Success
and “Up to date with main” notices disappear after 30 seconds. If GitHub's
comparison service is unavailable for an archive update, the notice reports
that the count is unavailable; the source update still completes.

Git checkouts use a fast-forward pull and require tracked files to be clean.
Without Git, the console downloads public source archives using Python and
updates the Dagger folder only. It preserves local changes when upstream has
not changed those files, and stops if changes overlap. New local files,
browser-saved exercise edits, `results/`, `dev/`, and credentials are preserved.
Archive updates leave Git history untouched; continue using the archive
updater or reconcile that checkout before returning to Git. An offline check
shows an error and leaves the installed console usable.

Open the local URL printed by the command (usually http://127.0.0.1:8767).
Choose **Odo relay · GPUs** or **AWS CPU · personal VM**, paste the workshop
password, and click **Connect**. The password field clears after submission.
For CPU mode the page shows VM startup/preparation progress, or a queue when
capacity is full. Once Ready appears, run the demos and exercises normally.
GPU controls are available through Odo; CPU mode offers CPU demos and MPI
ranks sharing the assigned VM's cores.

Click **Disconnect** when finished. Stopping the local server also closes its
SSH connection and releases the CPU VM. Closing only the browser tab leaves
the local server and its SSH connection running; reopen the page to return.
Your local exercise edits remain, while files stored only on a released CPU
VM are discarded.

## Instructor preparation

The deployment uses `~/awsmcp` and the instructor's `lab` profile locally.
It gives the coordinator an EC2 IAM role for worker lifecycle operations, so
its ongoing operation does not depend on an expiring personal SSO session.
The gateway remains available until explicit cleanup; CPU workers have a
three-hour maximum lifetime and a 30-minute inactivity timeout.

```bash
# Create or update the gateway. Updating restarts the coordinator and ends
# active console sessions, so do this before attendees connect.
aws/gateway deploy

# Prepare a clean Julia CPU image once. New workers reuse its packages.
aws/gateway bake

# Authenticate Odo once, in the instructor's terminal.
aws/gateway connect-odo

# Inspect service health, current leases, and recent log entries.
aws/gateway status
```

Deployment writes the **private** password handout to
`dev/gateway-access.txt` and its management state to
`dev/gateway-deployment.json`, both outside Git. Give attendees the workshop
password from that handout. Publish the console changes and the **public**
`host/workshop.json` descriptor together. The descriptor contains no password,
AWS credentials, or private SSH key.

`connect-odo` prompts for the instructor's Odo password, or can use an existing
local helper via `--odo-askpass PATH`. Authentication material is placed only
in a root-private RAM directory and removed after authentication. The Odo
connection is kept by a separate service; restarting the CPU coordinator does
not require reauthenticating the global Odo connection. If Odo disconnects,
rerun `connect-odo`; affected attendees reconnect their consoles. CPU mode is
available while Odo is offline.

Only the instructor runs the `aws/gateway` commands. Attendees run the local
console launcher. Use `--gateway FILE` for an alternate descriptor; a raw host
also requires `--gateway-host-key 'ssh-ed25519 ...'` and optionally
`--gateway-port PORT`.

## Allocation and connection lifetime

One console connection owns one CPU VM. Uploading the tutorial, preparing a
workspace, running demonstrations, and cancelling runs share that connection;
individual commands do not allocate additional VMs. The coordinator assigns
a fresh session ID to each connection, so a shared password does not make
attendees share CPU VMs or results.

The coordinator counts pending/running Standard EC2 vCPUs plus reservations
for launches not yet visible to EC2. Admission is serialized and queued in
arrival order under the **256-vCPU budget**, including the coordinator and
other running instances. With the existing 8-vCPU instance and the 4-vCPU
coordinator, 30 eight-vCPU workers use **252 vCPUs**.

Disconnecting terminates that connection's worker and its security group.
The slot remains reserved until cleanup finishes. An interrupted connection
is detected through SSH keepalives (15 seconds, three unanswered checks).
Brief network interruptions can recover while the existing connection is
still alive. Once it is closed, reconnecting obtains a fresh VM. Idle and
maximum-lifetime checks also close sessions. CPU workers have an independent
shutdown watchdog if the coordinator fails.

The coordinator persists lease ownership, retries failed cleanup, and reaps
orphaned workers with this gateway's exact ownership tag prefix. On restart,
it cleans up leases from the previous process before admitting new ones.
Resources belonging to other fleets are excluded from that recovery.

## Authentication boundary

The shared password authenticates the dedicated `workshop` SSH account. SSH
forces that account into the gateway handler, allows one session per transport,
and disables shell commands, PTYs, SFTP, TCP/socket/agent forwarding, and user
startup scripts. Only `cpu` and `odo` backend requests are accepted. Commands
and uploaded code execute on a CPU worker or on Odo, never on the coordinator.

The local server accepts password submission only from localhost, uses its
per-launch HTTP token, and passes the password to OpenSSH through a private
Unix socket. The password is absent from argv, environment variables, files,
application logs, and browser storage. Gateway SSH identity is pinned in the
public descriptor. The gateway verifies the password using its normal salted
system password hash.

Only gateway port 22 is exposed to attendees. Each CPU worker's SSH security
group admits the gateway's address only; its per-worker private key stays on
the coordinator, and workers have no coordinator IAM role. Attendee IP
whitelists and key enrollment are unnecessary. The Odo path uses the existing
single-transport relay and shared Odo identity, with separate session folders
and Slurm job names.

## Cleanup

After the workshop:

```bash
aws/gateway cleanup
```

This stops the coordinator, releases active workers, terminates the gateway,
and removes its IAM role/profile and prepared image/snapshot. Keep the private
management state until cleanup completes. The public descriptor becomes stale
when the gateway is removed; a new deployment regenerates it.
