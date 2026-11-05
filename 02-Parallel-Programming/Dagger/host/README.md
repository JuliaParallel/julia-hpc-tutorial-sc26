# Local tutorial console

From `02-Parallel-Programming/Dagger` on your **own computer**:

```bash
python3 host/app.py
```

On first run it asks for your OLCF username, adds or confirms one marked SSH
host entry, opens the SSH connection, and then serves the page. See
[Credentials](#credentials) for what it writes and how an existing `odo` entry
is treated.

Open **http://127.0.0.1:8767**, then select **Connect & allocate**, then
**Run demo** in any console. Requires
Python 3.10+ and OpenSSH with connection multiplexing locally; no Python
packages, local Julia, Node, or internet-hosted web assets are needed. Odo's
Python 3.6 runs only the coordinator.

The console uses your SSH alias `odo`. With the existing `odo-connect` helper,
authentication happens automatically through that helper. Otherwise SSH keys
or an agent must already work; for password/MFA authentication, open the
console's dedicated transport once from a local terminal:

```bash
mkdir -p ~/.ssh/control
ssh -S ~/.ssh/control/sc26-portal -MNf odo
```

Passwords never pass through the webpage. The dedicated connection avoids
Odo's one-session-per-transport limit while another terminal is connected.

## Credentials

The console stores no credentials. Your Odo identity lives only in your SSH
configuration, agent, and control socket — never in this repository, the
webpage, or the uploaded workspace. Startup automates the setup:

1. It resolves an identity from `--odo-user`, `SC26_ODO_USER`, or an existing
   working `odo` alias. If none is available and a terminal is attached, it
   asks: `OLCF/UCAMS username for Odo [...]`. The bracketed default is the
   username already in your SSH config, otherwise your local user.

2. If an alias already points at `login1.odo.olcf.ornl.gov`, the console uses
   it unchanged. Otherwise it owns exactly one marked block in
   `~/.ssh/config` and creates `Host sc26-odo-<user>`:

   ```
   # >>> sc26-dagger console: managed host block >>>
   Host sc26-odo-YOUR_ODO_USERNAME
       HostName login1.odo.olcf.ornl.gov
       User YOUR_ODO_USERNAME
       ServerAliveInterval 60
       ControlMaster auto
       ControlPath /home/you/.ssh/control/sc26-portal
       ControlPersist 8h
   # <<< sc26-dagger console: managed host block <<<
   ```

   The block goes at the **top** of the file because OpenSSH keeps the first
   value it obtains for `User`, `HostName`, `Port` and `ControlPath`,
   regardless of block specificity. After writing, the console verifies the
   alias with `ssh -G` and refuses to continue if anything resolved
   differently.

3. It opens the multiplexed connection (`ssh -S … -MNf`) and lets SSH prompt
   for your XCAMS/UCAMS password in that terminal. Odo needs no RSA token.
   Later traffic runs in `BatchMode` over that socket.

4. It derives the default workspace from the resolved username:
   `/gpfs/wolf2/olcf/trn047/scratch/<odo-user>/sc26-dagger/portal`. This fixes
   the old behaviour, which used your *local* username.

Passwords never pass through the webpage. Keep the control socket private:
`~/.ssh/control` is created with mode `0700`, and each launch prints a random
`X-Console-Token` that the browser must send for every command; unrecognized
host names are rejected too. Do not share the printed Tailscale URL with
anyone else: whoever opens it can submit work as you.

Skip the startup login with `--no-auth` (Connect then needs an existing
connection). Use `--odo-user NAME` to avoid the prompt, `--ssh-config PATH` to
manage a different file, and `--ssh-host NAME` to force an alias;
`--remote-root` remains available.

### If you already have an `odo` entry

- **Reused, not rewritten:** a working `Host odo` (or `--ssh-host`) that
  points at Odo and sets a `User` is used as-is when it matches the requested
  identity.
- **Never touched:** other `Host` blocks, `Include`d files, keys, and
  `known_hosts`. Your `odo-connect` helper keeps working.
- **Only one managed block:** it is replaced in place, left byte-for-byte
  unchanged when nothing changed, and the file is written atomically with mode
  `0600`.
- **Refused:** a hand-written `Host sc26-odo-<user>` (or your `--ssh-host`)
  outside the markers, a damaged marker pair, or a resolved mismatch after
  writing. Rename the entry or pass `--ssh-host`.
- **Shared, not duplicated:** the same hostname means the same `known_hosts`
  entry, and the socket directory exists once. Two consoles share the master
  socket; each still owns and cancels only its own Slurm job.
- **Possible leftovers:** `ControlPersist 8h` keeps the master after the
  console exits. Close it with `ssh -O exit sc26-odo-<user>`. An existing
  master on the same socket is reused; a stale one fails with SSH's own
  message.
- **Global options:** a `Host *` `User` or `HostName` cannot override the
  managed block because it comes first. A `Host * ControlPath` does not affect
  console traffic either, since every command passes `-S` explicitly; it only
  affects a manual `ssh sc26-odo-<user>`.

On connection the console uploads the current tutorial code to its own marked
scratch workspace, prepares packages on the **login node**, and submits a
one-node holding job. It waits for that job to start before enabling Run.
The default workspace is
`/gpfs/wolf2/olcf/trn047/scratch/<odo-user>/sc26-dagger/portal`.
Existing unmarked, nonempty directories are rejected. Reconnecting refreshes
course code there; results and Julia preferences stay put. This is separate
from your normal checkout and exercise edits.

Override your remote username/path, account, reservation, or time limit:

```bash
python3 host/app.py --remote-root /gpfs/wolf2/olcf/trn047/scratch/YOUR_ODO_USER/sc26-dagger/portal \
    --account trn047 --time 02:00:00
```

The server listens on **`0.0.0.0:8767`** by default, so you can also open it
from your laptop over Tailscale. Startup prints the machine's Tailscale URL
when the Tailscale CLI is available. Tailscale IPs and the machine's MagicDNS
name are accepted; `--allow-host NAME` adds another browser hostname.
Use `--host 127.0.0.1` for a loopback-only server.

Other options: `--odo-user`, `--ssh-host`, `--ssh-config`, `--control-path`,
`--account`, `--reservation`, `--time`, `--port`, `--no-auth`, and
`--no-browser`. `python3 host/app.py --help` lists them.

## What stays alive

The webpage and plot renderer run locally. A login-side Python coordinator
owns one Slurm job; a sleeping batch script retains that allocation while
`srun` steps perform the actual work. A Julia worker retains loaded packages,
compiled demo methods, and REPL variables between commands in the same layout.

- Single-process demos use one Julia process with four visible GPUs (the
  multi-GPU algorithm selects two or four).
- MPI demos use two ranks on that same node, with two GPUs available per rank.
  CPU MPI tasks still run on CPU scopes.
- Changing between those layouts stops the old step and starts a new worker.
  REPL variables reset; plot history remains in the local console.
- REPL expressions run in a persistent Julia module on Odo. Under MPI, the
  expression runs on **every rank**. Use identical graph-building decisions
  everywhere, just as in the demos. This is a Julia evaluator with multiline
  input, not a full terminal REPL: terminal modes such as `]`, `?`, shell mode,
  and interactive `readline` prompts are not supported.

**Cancel run** terminates the current step and clears its Julia workspace;
the allocation remains available. **Release node**, local Ctrl-C, or SSH EOF
terminates the worker and cancels only the job submitted by this console.
Closing the browser tab leaves the local app and its allocation running;
reopen the page to recover the session and its recent history. A machine or
network failure that prevents remote cleanup may leave the holding job until
its Slurm time limit; its job ID appears in the session output/status.

## Source and plots

Every demo uses Highlights.jl's Tree-sitter Julia parser and a Dracula theme
to print syntax-highlighted excerpts from the actual source files,
then runs with Dagger logging and prints a UnicodePlots Gantt. `NO_COLOR` or
`SC26_COLOR=0` disables terminal source color. This also works with the
existing terminal launchers and `include("demos/01_tasks.jl")`.
Function calls, macros, types, numbers, strings and comments receive distinct
colors. The webpage preserves the same 24-bit colors. The Julia grammar and
theme artifacts are downloaded on the login node during preparation.

The algorithm panel updates immediately when you select a demo, including its
CPU/GPU and MPI settings. Selecting a past report restores its executed source
snapshot. REPL output and late events from another demo do not replace the
selected algorithm. Developers should regenerate the bundled syntax highlighting
after editing algorithm excerpts with `julia --project host/build_sources.jl`;
the console always reads the current source and falls back to plain text if the
highlighted catalog is stale.

The web console adds processor lanes with immediate task tooltips, a transfer toggle,
zoom, a function-time summary, and the last 20 timelines for comparison.
Every demo has its own console — Algorithm, Julia REPL, Task timeline and
Session output — and runs are attributed to the console that started them.
The sidebar groups **01 CPU** (Tasks & dependencies, Mutable data & tiles) and
numbers the remaining demos 02–06. Choosing a sidebar entry shows that section
alone in the main area: the CPU page holds both consoles, every other page is
one console. The section-level block is titled **Demo**, since the page heading
already carries the section name. The last viewed page and each panel's collapsed state are
remembered per browser. Within a console the Algorithm and Julia REPL panes
collapse independently, keeping their width and shrinking vertically to the
title bar; on narrow screens they stack.

For the CPU Datadeps, GPU, MPI, and multi-GPU demos, the section page shows two
sibling panels: **Demo** and a collapsible **Exercise** panel. The Exercise
panel proposes the step for that target and contains an editor for
`my_cholesky!`. The editor is shared by all four exercise sections, so code
written for the CPU run is reused unchanged on the GPU and MPI runs.
**Run exercise** writes the editor contents to the remote workspace and runs
`harness/run.jl` with the selected target and worker layout; **Reset code**
restores the student stub. Results and timelines appear in the Exercise panel's
own output and timeline panels. A collapsible **Setup and Use** primer contains
an editable, incomplete setup template: create the matrix, tile it with
`DArray(A, ...)`, place it with the target scope, and call
`my_cholesky!(...)`. Fill in the blanks or use it as a reference; **Run
exercise** performs the same setup in the harness. The **Function** panel holds
the `my_cholesky!` editor and the Reset/Run buttons.

**Save JSON** downloads the selected report. Reports also remain in the Odo
workspace under `results/demo-timelines/` for ordinary terminal runs.

Timelines show **host-side compute and movement events**, including compilation.
They are not GPU kernel profiles or benchmark results. MPI events are combined
from both ranks and use seconds since each rank's post-barrier logging start;
this is approximate alignment across physical hosts, not clock synchronization.
The workshop uses one host. Repeat a demo to see warm execution.

Compute lanes identify the MPI rank that actually executes the function;
scheduling wrappers logged by other ranks are omitted. Both ordinary movement
events and `move!` tasks are classified as data movement, excluded from the
compute summary. Datadeps transfers show source → destination ranks, with exact
CPU/GPU memory endpoints in the tooltip. Each transfer interval also identifies
its recording rank: the sending and receiving ranks can both log their part of
the same transfer. When Dagger does not record a source endpoint, the tooltip
says so rather than inferring it from the recording rank.

The backend can also be used directly from Julia:

```julia
include("demos/presentation.jl")
Dagger.enable_logging!(all_task_deps=false, taskfuncnames=true,
                       taskuidtotid=true, timeline=true)
# Run and fetch your Dagger tasks here.
logs = Dagger.fetch_logs!()
Dagger.disable_logging!()
Dagger.show_logs(stdout, logs, :unicode_gantt)
```

## Checks

```bash
python3 -m unittest discover -s test -p 'test_*.py'
julia --project -t 4 test/presentation.jl
node test/ansi.cjs  # optional developer checks; Node is not needed to use the app
node test/timeline.cjs
```

These check the local protocol, allocation cleanup, HTTP command boundary,
source extraction, event timing and UnicodePlots backend without Odo. Live
CPU/GPU/MPI validation is recorded in `../PLAN.md`.
