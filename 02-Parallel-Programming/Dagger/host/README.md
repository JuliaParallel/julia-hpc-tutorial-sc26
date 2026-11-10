# Local tutorial console (dry run on andromeda)

This branch, `fda/dry-run`, points the console at **andromeda**
(`andromeda.lasdpc.icmc.usp.br`, SSH port 21200, ICMC/USP): one Slurm node,
`tau01`, with 48 cores and two NVIDIA A30 GPUs in partition `gpu`. The GPU
demos and exercises therefore use CUDA (`Dagger.scope(cuda_gpu=1)`, cuBLAS,
cuSOLVER) where the Odo course uses AMDGPU.

From `02-Parallel-Programming/Dagger` on your **own computer**:

```bash
python3 host/app.py
```

or, without a checkout, with the course installer:

```bash
curl -fsSL https://fda-tome.github.io/sc26/install.sh | SC26_REF=heads/fda/dry-run SC26_HOME=~/sc26-dry-run sh
```

On first run it asks for your andromeda username, adds or confirms one marked
SSH host entry, opens the SSH connection with your SSH key (andromeda takes no
password), and serves **http://127.0.0.1:8767**. Requires Python 3.9+ and
OpenSSH with connection multiplexing locally; no Python packages, local Julia,
Node, or internet-hosted web assets are needed.

## No allocation: one Slurm job per run

There is no node to allocate or release. **Connect** uploads the tutorial
code to `~/sc26-dagger/portal`, downloads Julia 1.13.1 and the packages on the
login node (`andromeda/prepare.sh`; the first time takes a few minutes), and
compiles them in one Slurm job with 16 CPUs and a GPU
(`andromeda/precompile.sh`). The page is ready when that job ends.

Every **Run demo**, **Run exercise**, **Run final exercise** and REPL
**Evaluate** is then submitted as its own Slurm job, with the resources that
run needs (`andromeda/launch.sh`), and the job ends with the run:

| Run | Slurm job |
|---|---|
| Julia basics, CPU, Multi-node · CPU MPI demos | 8 CPUs, no GPU (MPI: two ranks of 4 CPUs) |
| Single GPU, Stencils demos | 8 CPUs, one GPU |
| Multiple GPUs demo | 8 CPUs, both GPUs |
| Multi-node + multi-GPU demo | two ranks, 4 CPUs and one GPU each |
| Exercises and final exercise | from Target and Processes: CPU no GPU, GPU and Mixed one, All GPUs both; under two MPI ranks every GPU target gives each rank its own GPU |
| REPL evaluation | the section demo's GPUs, with the REPL's own layout |

Each job asks for 4 GB per CPU and the time limit `--time` (default 20
minutes). The status line follows the job: submitted, queued (with its job
ID) when the node is busy, running, done. **Cancel run** stops the run and its
job. **Disconnect**, local Ctrl-C or a dropped SSH connection does the same
for a run in progress; nothing else is held. SSH probes the connection every
30 seconds and gives it five minutes; the coordinator writes a heartbeat
every 30 seconds, so after a lost connection it notices and stops its run, and
the next **Connect** also stops any coordinator an earlier connection left
behind (`results/agent.pid`).

Every run is a fresh Julia process: REPL variables do not carry over from one
evaluation to the next, and every timeline includes compilation. A CPU demo
takes about 10–20 seconds, a GPU demo somewhat longer, and the final
exercise about a minute on the CPU.

Two MPI ranks are two Slurm tasks of one job, started by `srun --mpi=pmi2`
(MPICH speaks PMI-2; it refuses PMIx), and Slurm binds each rank's CPUs and
GPU. `SC26_MPI_LAUNCH=mpiexec` instead runs one task that holds both ranks'
CPUs and GPUs, with MPICH's own `mpiexec` starting the ranks inside it
(`andromeda/mpirun.jl`; `andromeda/rank.sh` keeps one GPU visible per rank).
`--launcher local` always uses that second way.

Your andromeda account needs a Slurm association (`sacctmgr show assoc
user=$USER`): with `AccountingStorageEnforce=associations`, a user without
one gets "Invalid account or account/partition combination specified" for
every job. Until an administrator adds it, `--launcher local` runs each
command directly on the node, without Slurm, for testing only.

Options: `--user` (also `SC26_USER`), `--ssh-host`, `--ssh-config`,
`--control-path`, `--partition` (default `gpu`), `--account`,
`--reservation`, `--time` (per run), `--remote-root` (absolute, or relative to
your remote home), `--launcher`, `--port`, `--host`, `--allow-host`,
`--no-auth`, `--no-auto-connect` and `--no-browser`.
`python3 host/app.py --help` lists them.

## Credentials

The console stores no credentials. Your identity lives only in your SSH
configuration, agent, keys and control socket, never in this repository, the
webpage, or the uploaded workspace.

1. It resolves a username from `--user`, `SC26_USER`, or an existing working
   `andromeda` alias. Otherwise it asks:
   `Username on andromeda (andromeda.lasdpc.icmc.usp.br) [...]`.

2. If an alias already reaches `andromeda.lasdpc.icmc.usp.br` on port 21200,
   the console uses it unchanged. Otherwise it owns exactly one marked block at
   the top of `~/.ssh/config`:

   ```
   # >>> sc26-dagger dry-run console: managed host block >>>
   Host sc26-andromeda-YOUR_USERNAME
       HostName andromeda.lasdpc.icmc.usp.br
       Port 21200
       User YOUR_USERNAME
       ServerAliveInterval 30
       ServerAliveCountMax 10
       ControlMaster auto
       ControlPath /home/you/.ssh/control/sc26-andromeda
       ControlPersist 8h
   # <<< sc26-dagger dry-run console: managed host block <<<
   ```

   The markers differ from the Odo console's, so both blocks can live in one
   file without touching each other. The block goes first because OpenSSH
   keeps the first value it obtains for `User`, `HostName`, `Port` and
   `ControlPath`. After writing, the console checks the alias with `ssh -G`.

3. It opens the multiplexed connection (`ssh -S … -MNf`) with your key; a key
   passphrase can be typed in that terminal. Later traffic runs in `BatchMode`
   over that socket. `ControlPersist 8h` keeps it after the console exits;
   close it with `ssh -O exit sc26-andromeda-<user>`.

Each launch prints a random `X-Console-Token` that the browser must send with
every command, and unrecognized host names are rejected. Do not share the
printed Tailscale URL: whoever opens it can submit work as you.

The upload goes to a dedicated, marked workspace (`~/sc26-dagger/portal` by
default); an existing unmarked, nonempty directory is refused. Reconnecting
refreshes the course code there; results and Julia packages stay put. The
server listens on `0.0.0.0:8767`, so you can also open it over Tailscale; use
`--host 127.0.0.1` for a loopback-only server.

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
Session output — and runs are attributed to the console that started them:
a run's output, timeline and source go to the Demo, Exercise or Final Exercise
panel that started it. The sidebar groups **02 CPU** (Tasks & dependencies,
Mutable data & tiles); Julia basics is 01, the other demos are 03–07 with
**07 Stencils** last, and **08 Final Exercise** closes the list. Choosing a
sidebar entry shows that section alone in the main area; the page heading
names the section. The last viewed page and each panel's collapsed state are
remembered per browser. Within a console the Algorithm and Julia REPL panes
collapse independently, keeping their width and shrinking vertically to the
title bar; on narrow screens they stack.

Every editor — REPL, Function, Setup and Use, Simulation — highlights Julia
as you type, in the Dracula colours of the demo sources (`host/highlight.js`:
a small tokenizer for keywords, calls, types, macros, strings, numbers,
symbols and comments; `???` placeholders are marked). The textarea stays the
editor; the coloured copy is drawn behind it, so selection, undo and
Ctrl/⌘+Enter work as before.

Every section page after Julia basics shows two sibling panels: **Demo** and
a collapsible **Exercise** panel. The Exercise panel's head has two controls: **Target**
(CPU, GPU, All GPUs, or Mixed) and **Processes** (One process or Two MPI
ranks). Each section starts on its own choice, for example GPU in Single GPU
and Two MPI ranks in the MPI sections. Inside the panel are the section's
prompt and two editors:

- **Setup and Use** is runnable code: it chooses the Dagger scope for the
  target, builds a seeded test input, tiles it with `DArray`, calls the
  Function, and checks the result (it prints `PASS`). Under MPI every rank
  runs it and only rank 0 prints. Changing Target or Processes loads the
  matching template, after asking if you have edited it; **Reset setup**
  restores it. Each section keeps its own Setup and Use.
- **Function** holds the code you write. In the Cholesky sections (CPU to
  Multi-node + multi-GPU) it is `my_cholesky!`, shared by all of them, so code
  written for the CPU run is reused unchanged on the GPU and MPI runs. The
  **Stencils** section has its own Function, `my_heat!`
  (`exercise/my_stencil.jl`): explicit heat diffusion on a periodic plate with
  `@stencil` and `@neighbors(u[idx], 1, Wrap())`, checked against the same
  steps in plain Julia. **Reset code** restores the student stub.

**Run exercise** runs the Function, then Setup and Use, in a fresh Julia
process, as one Slurm job with the resources Target and Processes ask for.
The terminal harness (`andromeda/run.sh RANKS GPUS [options]`, which calls
`harness/run.jl`) still offers `--strict`, size sweeps, and CSV results.

The final sidebar entry, **Final Exercise**, is a Gray-Scott
reaction-diffusion experiment on a periodic grid that uses both earlier
exercises. u and v are DArrays split into panels of whole columns. Each ADI
half step computes the explicit terms — diffusion along y and the reaction —
with one `@stencil` block, solves the implicit direction column by column with
the Cholesky factor of `I - (dt/2) D L` (one `LAPACK.potrs!` task per panel;
cuSOLVER on a GPU task), and transposes u and v, so calling the same half step
again treats the other direction implicitly. It has the same Target and
Processes controls and the same **Setup and Use** editor as the other
exercises: the **Simulation** editor (`results/final-exercise/final_exercise.jl`)
defines `simulate!(u, v, p, nsteps)`, and Setup and Use sets the parameters
(`dt = 2`, 1800 steps), the initial fields and the panels, calls it where the
scope says, and checks the result. The solution, `solution/final_exercise.jl`,
matches a serial ADI to round-off on the CPU, one GPU, both GPUs, and two MPI
ranks with and without GPUs. A workspace file from an older console is kept
aside: the one-file layout as `final_exercise.old.jl`, the previous
stencil-free `simulate!` as `final_exercise.pre-stencil.jl`. Above the task
timeline, a **Visualization** panel shows the substrate u and activator v
fields side by side, with a slider and Play button that replay the sampled
time steps. The worker gathers those fields after a successful run and the
console draws high-resolution canvas plots, so the plotting code stays out of
the submitted script. The panel also reports the F and k parameters. The field
checkboxes show or hide either picture immediately; no rerun is needed. The
same page has a collapsible **Use your agents** panel with OpenCode, Claude,
and Codex buttons. Each opens an embedded terminal in the page and runs the
agent in a local PTY on the machine running the console (not on andromeda).
The agent must be on `PATH`; the terminal endpoints accept requests only from
localhost. If an agent is missing, the panel asks for permission and runs its
installer: OpenCode and Claude use their install scripts, and Codex uses npm
or Homebrew. The embedded terminal uses vendored xterm.js and the fit addon
(MIT); see `xterm-LICENSE` and `addon-fit-LICENSE`.

**Save JSON** downloads the selected report. Reports also remain in the
andromeda workspace under `results/demo-timelines/`.

Timelines show **host-side compute and movement events**, including compilation.
They are not GPU kernel profiles or benchmark results. MPI events are combined
from both ranks and use seconds since each rank's post-barrier logging start;
this is approximate alignment across physical hosts, not clock synchronization.
andromeda is one host. Every console run is a fresh Julia, so its timeline
includes compilation.

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

These check the local protocol, the per-run job lifecycle (one command per
job, Cancel and EOF ending it, a job that dies early reported as a failed
run), the HTTP command boundary, source extraction, event timing and the
UnicodePlots backend, without a cluster.
