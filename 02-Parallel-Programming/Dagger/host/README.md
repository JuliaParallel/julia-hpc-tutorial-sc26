# Local tutorial console

From `02-Parallel-Programming/Dagger` on your **own computer**:

```bash
python3 host/app.py
```

Open **http://127.0.0.1:8767**, then select **Connect & allocate**. Requires
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

On connection the console uploads the current tutorial code to its own marked
scratch workspace, prepares packages on the **login node**, and submits a
one-node holding job. It waits for that job to start before enabling Run.
The default workspace is
`/gpfs/wolf2/olcf/trn047/scratch/<local-user>/sc26-dagger/portal`.
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

Other options: `--ssh-host`, `--control-path`, `--reservation`, `--port`,
and `--no-browser`. `python3 host/app.py --help` lists them.

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
