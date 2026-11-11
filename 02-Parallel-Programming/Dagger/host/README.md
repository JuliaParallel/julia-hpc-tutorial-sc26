# Local tutorial console

From `02-Parallel-Programming/Dagger` on your **own computer**:

```bash
python3 host/app.py
```

On first run it asks for your OLCF username, adds or confirms one marked SSH
host entry, opens the SSH connection, and then serves the page. See
[Credentials](#credentials) for what it writes and how an existing `odo` entry
is treated.

Open **http://127.0.0.1:8767**; the console immediately prepares the Julia
packages and compiles them in a short Slurm job, and enables the buttons once
the page says it is ready. Nothing is allocated in between: every run is its
own Slurm job (see [One Slurm job per run](#one-slurm-job-per-run)). Pass
`--no-auto-connect` to wait for **Connect** instead. Requires
Python 3.9+ and OpenSSH with connection multiplexing locally; no Python
packages, local Julia, Node, or internet-hosted web assets are needed. Odo's
Python 3.6 runs only the coordinator.

Every console opens its own SSH connection when it starts and asks for your
PASSCODE in its terminal; consoles never share a connection. Passwords never
pass through the webpage, and a console's own connection avoids Odo's
one-session-per-transport limit while another terminal is connected. To reuse
a connection you opened yourself instead, pass its socket with
`--control-path`.

### Several consoles at once

Run the console as many times as you like, for one Odo user or several. Each
one has its own SSH connection (`~/.ssh/control/sc26-<user>-<random>`), closed
when that console exits; its own page, on the first free port from 8767 (the
second console prints `http://127.0.0.1:8768`, and so on); its own session
folder and Slurm job on Odo; and, per Odo user, its own Simulation file
(`results/final-exercise/<user>/`). The first time a user starts, that file is
copied from the one older consoles kept for everyone. To start one for another
user: `python3 host/app.py --odo-user OTHER_USER`, or with the installer
`curl -fsSL https://fda-tome.github.io/sc26/install.sh | sh -s -- --odo-user OTHER_USER`.

## Credentials

The console stores no credentials. Your Odo identity lives only in your SSH
configuration, agent, and control socket — never in this repository, the
webpage, or the uploaded workspace. Startup automates the setup:

1. It resolves an identity from `--odo-user`, `SC26_ODO_USER`, or an existing
   working `odo` alias. If none is available and a terminal is attached, it
   asks: `OLCF/UCAMS username for Odo [...]`. The bracketed default is the
   username already in your SSH config, otherwise your local user.

2. If an alias already points at `login1.odo.olcf.ornl.gov`, the console uses
   it unchanged. Otherwise it owns one marked block per user in
   `~/.ssh/config` and creates `Host sc26-odo-<user>`:

   ```
   # >>> sc26-dagger console: sc26-odo-YOUR_ODO_USERNAME >>>
   Host sc26-odo-YOUR_ODO_USERNAME
       HostName login1.odo.olcf.ornl.gov
       User YOUR_ODO_USERNAME
       ServerAliveInterval 60
   # <<< sc26-dagger console: sc26-odo-YOUR_ODO_USERNAME <<<
   ```

   The block names no shared socket: each console passes its own. A new
   block goes at the **top** of the file because OpenSSH keeps the first
   value it obtains for `User`, `HostName` and `Port`, regardless of block
   specificity; an existing one is updated where it is. The single block of
   older consoles is replaced. After writing, the console verifies the
   alias with `ssh -G` and refuses to continue if anything resolved
   differently.

3. It opens the multiplexed connection (`ssh -S … -MNf`) and lets SSH prompt
   for your XCAMS/UCAMS password in that terminal. Odo needs no RSA token.
   Later traffic runs in `BatchMode` over that socket.

4. It derives where sessions live from the resolved username:
   `/gpfs/wolf2/olcf/trn047/scratch/<odo-user>/sc26-dagger/sessions`. This
   fixes the old behaviour, which used your *local* username.

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
- **One managed block per user:** each is updated in place, left
  byte-for-byte unchanged when nothing changed, and the file is written
  atomically with mode `0600`. Another user's block is never touched.
- **Refused:** a hand-written `Host sc26-odo-<user>` (or your `--ssh-host`)
  outside the markers, a damaged marker pair, or a resolved mismatch after
  writing. Rename the entry or pass `--ssh-host`.
- **Shared, not duplicated:** the same hostname means the same `known_hosts`
  entry. Connections are never shared: every console opens its own, and owns
  and cancels only its own Slurm job.
- **Possible leftovers:** a console closes its connection when it exits; one
  killed outright leaves it to close itself after 8 hours.
- **Global options:** a `Host *` `User` or `HostName` cannot override the
  managed block because it comes first. A `Host * ControlPath` does not affect
  console traffic either, since every command passes `-S` explicitly.

### One folder and one job name per connection

Many consoles can connect with the same SSH credentials, so every
**Connect** starts a session of its own. The console picks a unique name,
`sc26-<date>-<time>-<random>` (for example `sc26-20261111-163012-a1b2c3`), and
creates exactly that folder under
`/gpfs/wolf2/olcf/trn047/scratch/<odo-user>/sc26-dagger/sessions/`. The
folder is created with a plain `mkdir`, which fails rather than reuse a folder
that already exists, and is readable only by your account. Everything the
session does lives in it: the uploaded course code, its `Manifest.toml` and
MPI preferences, and the results and timelines. The same name is the name of
every Slurm job the session submits (`-J`), so `squeue --me` shows which job
belongs to which console, and Cancel, Disconnect or Ctrl-C cancels only that
session's job. The page's sidebar shows
the session's folder. A new Connect makes a new folder; earlier folders and
their results stay until you delete them.

The one thing sessions share is the Julia package depot,
`…/sc26-dagger/depot`, so a new session does not download and compile
everything again. Package preparation on the login node runs one session at a
time (a lock on `…/sc26-dagger/prepare.lock`), and each new session starts
from the manifest the last one resolved, so all sessions run the same package
versions even when Dagger's `master` moves. Sessions under one account are
kept apart by the console's own paths, not by the operating system: the same
account can read every one of its folders.

If the console cannot create the sessions directory, it says so: your project
scratch directory (`/gpfs/wolf2/olcf/<account>/scratch/<odo-user>`) is created
by OLCF for members of the project, so check `groups` on Odo. Pass `--account`
for another project, or `--remote-root DIR/sessions` for another directory you
can write on Odo; the package depot then lives in `DIR` too.

Then the session prepares packages on the **login node**, compiles them in a
short job (`frontier/precompile.sh`, which runs `frontier/setup.sh`), and
enables Run. This is separate from your normal checkout and exercise edits.

Override your remote username/path, account, reservation, or time limit:

```bash
python3 host/app.py --remote-root /gpfs/wolf2/olcf/trn047/scratch/YOUR_ODO_USER/sc26-dagger/sessions \
    --account trn047 --time 02:00:00
```

The server listens on **`0.0.0.0:8767`** (or the next free port) by default, so you can also open it
from your laptop over Tailscale. Startup prints the machine's Tailscale URL
when the Tailscale CLI is available. Tailscale IPs and the machine's MagicDNS
name are accepted; `--allow-host NAME` adds another browser hostname.
Use `--host 127.0.0.1` for a loopback-only server.

Other options: `--odo-user`, `--ssh-host`, `--ssh-config`, `--control-path`,
`--account`, `--reservation`, `--time`, `--port`, `--no-auth`, and
`--no-browser`. `python3 host/app.py --help` lists them.

## One Slurm job per run

The webpage and plot renderer run locally; a login-side Python coordinator
runs on Odo. There is no holding job: every **Run demo**, **Run exercise**,
**Run final exercise** and REPL **Evaluate** is submitted as its own Slurm job
(`frontier/launch.sh` with `srun`, outside any allocation), named after the
session, with your account, the reservation if you pass `--reservation`, and
the time limit `--time` (default 20 minutes). The job ends with the run.

- One-process runs ask for one task with 7 CPUs, plus four GPUs when they use
  GPUs. The Multiple GPUs demo asks for exactly the GPUs chosen in its menu:
  2, 4 or 8 (an Odo node has eight: four MI250X with two GCDs each).
- MPI runs put **each rank on its own Odo node**, with the node's 56 CPUs as
  Julia threads and, when the run uses GPUs, all eight of its GPUs. Choose 1
  to 32 ranks with the **Ranks** menu: next to Run demo in the Multi-node
  sections, next to Processes in the Exercise and Final Exercise panels when
  Processes is MPI ranks, and next to the REPL's layout menu.
- The status line follows the job: submitted, queued with its job ID while it
  waits for a node, running, done. Its output streams into the Session output
  of the panel that started it.
- Every run is a fresh Julia: REPL variables do not carry over from one
  evaluation to the next, and every timeline includes compilation. Under MPI
  the expression runs on **every rank**; use identical graph-building
  decisions everywhere, just as in the demos. This is a Julia evaluator with
  multiline input, not a full terminal REPL: terminal modes such as `]`, `?`,
  shell mode, and interactive `readline` prompts are not supported.

**Cancel run** cancels the run's job (`scancel` by the session's name).
**Disconnect**, local Ctrl-C, or a lost connection does the same for a run in
progress; nothing else is held. The coordinator writes a heartbeat every 30
seconds, so it notices a lost connection and cancels its run. Closing the
browser tab leaves the local app running; reopen the page to recover the
session and its recent history.

The terminal workflow is unchanged: inside an allocation from
`frontier/allocate.sh`, `frontier/launch.sh` still runs steps of that
allocation.

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

Every section page after Julia basics shows two sibling panels: **Demo** and
a collapsible **Exercise** panel. The Exercise panel's head has two controls: **Target**
(CPU, GPU, All GPUs, or Mixed) and **Processes** (One process or Two MPI
ranks). Each section starts on its own choice, for example GPU in Single GPU
and Two MPI ranks in the MPI sections. Inside the panel are the section's
prompt and two editors:

- **Setup and Use** is runnable code: it chooses the Dagger scope for the
  target, builds a seeded test matrix, tiles it with `DArray`, calls
  `my_cholesky!(DA.chunks)`, and checks the factor (it prints `PASS`). Under
  MPI every rank runs it and only rank 0 prints. Changing Target or Processes
  loads the matching template, after asking if you have edited it; **Reset
  setup** restores it. Each section keeps its own Setup and Use.
- **Function** holds `my_cholesky!`. It is shared by all sections, so code
  written for the CPU run is reused unchanged on the GPU and MPI runs; **Reset
  code** restores the student stub.

**Run exercise** runs the Function, then Setup and Use, in a fresh Julia module
on a worker with the chosen Processes layout (the console restarts the worker
when the layout changes). Results and timelines appear in the Exercise panel's
own output and timeline panels. The terminal harness (`frontier/run.sh`,
`harness/run.jl`) is unchanged and still offers `--strict`, size sweeps, and
CSV results.

The final sidebar entry, **Final Exercise**, opens a Gray-Scott
reaction-diffusion experiment on a periodic grid. It explains how to build the
1D periodic Laplacian, factor the two ADI diffusion operators with Cholesky,
and write the two half steps whose column and row solves are independent
Dagger tasks. It has the same Target and Processes controls and the same
**Setup and Use** editor as the other exercises: the **Simulation** editor
(`results/final-exercise/final_exercise.jl`) defines
`simulate!(u, v, p, nsteps)`, and Setup and Use sets the parameters and the
initial fields, calls it where the scope says, and checks the result. A
workspace file from an older console, which held the whole script, is kept as
`final_exercise.old.jl`. Above the task timeline, a **Visualization** panel shows the
substrate u and activator v fields side by side, with a slider and Play button
that replay the sampled time steps. The worker returns those fields after a
successful run and the console draws high-resolution canvas plots, so the
plotting code stays out of the submitted script and the console source. The
panel also reports the F and k parameters. The field checkboxes show or hide
either picture immediately; no rerun is needed. The same page has a collapsible
**Use your agents** panel with
OpenCode, Claude, and Codex buttons. Each opens an embedded terminal in the page
and runs the agent in a local PTY on the machine running the console (not on
Odo). The agent must be on `PATH`; the terminal endpoints accept requests only
from localhost. If an agent is missing, the panel asks for permission and runs
its installer: OpenCode and Claude use their install scripts, and Codex uses npm
or Homebrew. The embedded terminal uses vendored xterm.js and the fit addon
(MIT); see `xterm-LICENSE` and `addon-fit-LICENSE`.

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

These check the local protocol, the per-run job lifecycle, HTTP command boundary,
source extraction, event timing and UnicodePlots backend without Odo. Live
CPU/GPU/MPI validation is recorded in `../PLAN.md`.
