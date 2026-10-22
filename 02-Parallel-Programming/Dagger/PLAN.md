# SC26 Julia tutorial: Dagger.jl for HPC (1 hour)

This session follows the introduction to Julia and the parallel, portable GPU
programming section. It builds on GPU concepts with task dependencies, data
movement, and scheduling across CPU threads, GPU devices, and MPI ranks on one
node. It includes 30 minutes of teaching and demos and 30 minutes of hands-on work.

## Learning objectives

By the end of the session, participants should be able to:

- Create tasks with `Dagger.@spawn` and build dependencies by passing task results.
- Declare reads and writes with Datadeps: `In`, `Out`, and `InOut`.
- Expose parallelism by dividing a matrix into a grid of tiles with `DArray`.
- Run the same tiled algorithm on CPU threads, an AMD GPU, and MPI ranks.
- Check correctness and compare performance with the supplied harness.

The central idea: write one sequential-looking function, describe how its
tasks access data, and let Dagger schedule tasks and move data.

## Timeline

| Time | Block | What happens | Material |
|---|---|---|---|
| 0:00–3:00 | Why Dagger | Task-based scheduling across threads, GPUs, and ranks. | Introduction |
| 3:00–8:00 | Tasks | `Dagger.@spawn`, `fetch`, and implicit dependency graphs. | `demos/01_tasks.jl` |
| 8:00–16:00 | Datadeps and tiles | Mutating tasks with `In`/`Out`/`InOut`; `DArray`, `Blocks`, and `.chunks`. Work through read/write dependencies together. | `demos/02_datadeps.jl` |
| 16:00–20:00 | GPU scopes | Select an AMD GPU with `Dagger.scope(rocm_gpu=1)`. Explain data movement and BLAS dispatch. | `demos/03_gpu.jl` |
| 20:00–23:00 | Same-node multi-GPU | One Julia process partitions one matrix product across two or four distinct GPUs. | `demos/05_multigpu.jl` |
| 23:00–25:00 | MPI demos | The multi-node programming model: two ranks on one Odo node, then two GPUs per rank. The same Julia code runs across hosts. | `demos/06_multinode.jl`, `demos/07_multinode_multigpu.jl` |
| 25:00–30:00 | Exercise brief | Tiled Cholesky (POTRF/TRSM/SYRK/GEMM), its dependency graph, commands, and correctness checks. | `exercise/README.md` |
| 30:00–42:00 | Hands-on: CPU | Add Datadeps and task annotations, run the harness, and fix errors. | `exercise/my_cholesky.jl` |
| 42:00–50:00 | Hands-on: GPU | Run the same function with a GPU scope and compare results. | Harness with `--target=gpu` |
| 50:00–56:00 | Hands-on: MPI | Run on two ranks with `--mpi --strict` to check annotations across ranks. | Harness with `--mpi --strict` |
| 56:00–60:00 | Results and discussion | Plot results; discuss tile size, data movement, and portability. | `harness/plot.jl` |

## Presentation outline

1. Dagger.jl: one task model across processors.
2. HPC hardware: CPU cores, GPU devices, and the network.
3. Tasks and dependencies (`demos/01_tasks.jl`).
4. Mutation and why reads and writes must be declared.
5. Datadeps: `spawn_datadeps`, `In`, `Out`, and `InOut` (`demos/02_datadeps.jl`).
6. Tiles: `DArray(A, Blocks(bs, bs))` and `.chunks`.
7. GPU scopes and direct BLAS/LAPACK task dispatch (`demos/03_gpu.jl`).
8. Multiple GPUs on one server: one matrix product partitioned across distinct GPUs (`demos/05_multigpu.jl`).
9. Multiple servers: MPI/SPMD and uniform decisions across ranks (`demos/06_multinode.jl`), followed by MPI plus multiple GPUs per server (`demos/07_multinode_multigpu.jl`). Odo resource limits keep both ranks on one node; the Julia code also runs across nodes.
10. Tiled Cholesky: four kernels and a dependency graph for a 4×4 tile grid.
11. Exercise: edit the template, check correctness, then change execution targets.
12. Results: tile-size effects and comparison with built-in and vendor implementations.
13. Discussion: when task parallelism helps and how data movement affects performance.

## Hands-on checkpoints

| Checkpoint | Success signal |
|---|---|
| CPU | `bash frontier/run.sh 1 --impl=student` reports `PASS` for every trial. |
| GPU | `bash frontier/run.sh 1 --impl=student --target=gpu` reports `PASS`. |
| MPI | `bash frontier/run.sh 2 --impl=student --mpi --strict` reports `PASS` on two ranks. |
| Results | `julia --project harness/plot.jl` shows the passing configurations. |

`--strict` disables data-affinity placement so that incorrect mutation
annotations are more likely to fail across MPI ranks. A passing CPU run
alone does not establish that every `InOut` annotation is correct.

Participants who finish early can sweep tile sizes, compare `--impl=builtin`
and `--impl=vendor`, or explore `--target=mixed`. The vendor implementation
is a single-process baseline. The goal is to understand portability and
scaling; a task-based implementation need not outperform a vendor library
on a single device.

## Session preparation

- Confirm the resource layout below before assigning Odo allocation windows. Each attendee uses their own account, source copy, and results. Use `frontier/allocate.sh` only during the assigned window if access is rotated.
- `frontier/allocate.sh` downloads packages on the login node with `frontier/prepare.sh` before requesting resources. Then precompile on a compute node with `frontier/setup.sh`. For allocations obtained another way, run `prepare.sh` on the login node first. Prepare a shared course depot before attendees log in.
- Run `frontier/validate.sh` in a one-node allocation, including the four-rank GPU checks. Revalidate whenever the resolved Dagger master commit or ROCm module changes.
- Run each demo before the session and keep a warm Julia REPL for demos 1–3.
- Launch MPI in a separate terminal; release the REPL step first if it reserves the resources that MPI needs. Prepare recorded GPU and MPI demonstrations as fallbacks.
- Run `frontier/multinode.sh cpu` and `gpu` in the existing one-node allocation. They launch two MPI ranks, with two distinct GPUs per rank in the GPU case. The slides teach the multi-node model, but tutorial capacity on Odo keeps all ranks on one node. No extra presenter node is needed.
- If remote capacity requires the fallback below, preflight independent CPU work on attendees' laptops (including local MPI if used), or arrange a separate CPU service before the event. Odo login nodes are for editing and downloads, not class-wide computation.
- If GPU execution is unavailable, use the instructor demonstration or recordings for that checkpoint and continue independent CPU work on the preflighted local/alternate service.

## Materials

```text
README.md                   quickstart and environment notes
PLAN.md                     one-hour session guide
Project.toml                Julia dependencies
exercise/README.md          student handout
exercise/serial_cholesky.jl  runnable serial starting point
exercise/my_cholesky.jl      template to parallelize
solution/my_cholesky.jl      reference solution
harness/                    correctness checks, timing, CSV output, plotting
demos/                      tasks, Datadeps, GPU scopes, MPI
frontier/                   Odo environment, setup, allocation, launch scripts
```

## Resource plan for 25 attendees and one presenter

Every attendee works independently with their own account, implementation,
results, and one-node allocation. The presenter also uses one node. All
scheduled MPI examples use multiple ranks on that same node, including the
four-rank GPU placement demo. Reserve **26 healthy compute nodes** for the
session; the presenter keeps their allocation throughout.

The October 14, 2026 Slurm inspection found 30 batch nodes with eight GPU GCDs
per node, one drained node, a 12-hour partition limit, and no reservations.
That snapshot leaves at most 29 healthy nodes, so the 26-node layout has three
spare nodes if all other capacity is available. The account/user associations
allow four running jobs per user and 100 submitted jobs. The batch partition
reports `OverSubscribe=EXCLUSIVE`: independent jobs occupy whole nodes even
when using fewer GPUs. These are recorded settings, not a tutorial-day
availability guarantee. See the [Odo user guide](https://docs.olcf.ornl.gov/systems/odo_user_guide.html)
for architecture and storage.

| Independent session arrangement | Attendee nodes | Presenter nodes | Total | Assessment |
|---|---:|---:|---:|---|
| 25 simultaneous one-node jobs, exclusive policy | 25 | 1 | 26 | Recommended; supports CPU, multi-GPU, and same-node MPI |
| 25 simultaneous two-GPU sessions, sharing allowed | at least 7 | 1 | at least 8 | Optional capacity saving; not required for the tutorial |

The sharing bound counts GPU devices; CPU/memory placement and scheduling
still need rehearsal. The current allocator requests all eight GPUs and 56
physical CPU cores per node. Both requests would need revision for sharing.
Limiting `SC26_GPUS_PER_TASK` only changes a step inside the whole-node
allocation. The single-process matrix-product demo uses two GPUs by default,
with `--gpus=4` to demonstrate four devices. The four-rank MPI demo assigns one
GPU per rank, using four GPUs on one node.
Same-node MPI teaches SPMD, process communication, GPU placement, and strict
mutation checks. It does not exercise communication between physical hosts or
measure network scaling.

Slurm job steps run under the allocation owner's UID. Each attendee launches
under their own account; setting `SLURM_JOB_ID` does not let them join a
presenter's allocation. The presenter is never the execution service for
attendee code. See Slurm's [job launch guide](https://slurm.schedmd.com/job_launch.html).

### If node sharing is denied

Use the 26-node exclusive layout above. It supports all 25 attendees and the
presenter concurrently without rotating access or requiring laptop computation.
Keep the original 12-minute CPU, 8-minute GPU, 6-minute MPI, and 4-minute results
schedule. Attendees obtain one-node allocations during the talk and retain
these through the exercise. No policy change is needed, but a reservation for
26 healthy nodes is still necessary to guarantee access.

If fewer than 26 nodes can be guaranteed, use two waves of 13 and 12 independent
one-node sessions, retaining one presenter node: **14 nodes at peak**. A
proposed 30-minute fallback is ten minutes of local CPU work, a nine-minute
remote window for attendees 1-13, a nine-minute window for attendees 14-25,
then two minutes of results. Attendees outside their remote window continue
independent CPU/MPI checks, editing, or analysis on preflighted laptops or a
separately arranged CPU service. Each window covers GPU and same-node MPI
checkpoints. Rehearse compilation and handoffs before adopting this shorter
remote experience; this wave schedule has not been tested at class scale.

For rotating access, bound each allocation with the agreed `SC26_TIME`, for
example `SC26_TIME=00:09:00 bash frontier/allocate.sh`, and use the event's
`SC26_RESERVATION`. The default two-hour allocation is for continuous access.
Time limits start when allocations are granted, so queue delays shift handoffs.
Do not compute or precompile the class's workloads on Odo login nodes.

Short per-attendee batch jobs on one node are another fallback: everyone edits
independently and receives their own logs/results, with queueing and cold
startup in the turnaround. The existing `frontier/batch.sh` runs one harness
configuration per submission; a combined checkpoint runner would be additional
work. Without a compute reservation, fixed interactive windows are not
guaranteed. Preflight local CPU/MPI work and prepare recordings if remote
capacity cannot be secured. Continuous remote access requires 26 healthy nodes
under the current exclusive policy, or a validated alternative system.

Read `sinfo`, `scontrol show partition`, `sacctmgr show assoc`, and
`scontrol show reservation` before coordinating the event reservation. Do not
submit 25 speculative allocations to discover capacity. Confirm attendee
accounts, project storage permissions, and shared depot access, then rehearse
with a bounded subset of real attendee accounts. Prepare and precompile the
depot on compute nodes before the event. Disk precompilation does not eliminate
process-local GPU/MPI warm-up: the October 14 MPI GPU runs recorded 36-54 seconds
of cold warm-up. Rerun the single-node smoke suite, including four GPU ranks,
before freezing the course environment.

### What stays alive during interactive work

`frontier/allocate.sh` holds one Slurm allocation and its shell. It reserves the
node until the allocation ends, even when no computation is running. Exiting
Julia releases its job step; exit the allocation shell to release the node.
Slurm's [salloc documentation](https://slurm.schedmd.com/salloc.html) describes
this lifetime.

`frontier/repl.sh` keeps one Julia process alive, retaining loaded packages,
compiled functions/kernels, GPU initialization, and unchanged implementation
modules. Attendees edit their source and invoke complete harness checks again;
changed source reloads in a fresh module. The exercise does not require keeping
an unfinished computation or a particular matrix alive between checks.

`frontier/mpi-repl.sh 4` keeps four Julia ranks on the same node, their MPI
communicator, and their initialized runtimes alive. Rank 0 reads harness options
and broadcasts commands and source snapshots so every rank executes the same
check. `quit` releases all ranks. `frontier/run.sh` provides an alternative that
starts fresh Julia processes for each run, paying startup and warm-up again.

## Presenter workflow and visual feedback

Use the [Quarto deck](presentations/dagger/index.qmd) for the task graph, the
four Cholesky kernels, the hardware picture, and timed exercise checkpoints.
The opening connects to the preceding parallel, portable GPU programming
section; speaker notes give demo cues and pacing. Render with
`quarto render presentations/dagger/index.qmd` (see README.md for preview and
PDF instructions). Spend most of the session in a large-font SSH terminal with
three remote tmux panes: code, a warm compute-node Julia REPL, and results. Odo
currently permits one SSH session per transport, so put the panes inside one
SSH connection. Keep the login-node shell inside tmux so a network interruption does not lose the
allocation or launch process. After allocation, use
`tmux -L sc26 new-session -s tutorial` so new panes inherit the allocation
environment. Start the allocation and warm CPU/GPU examples
before the audience joins.

Use `frontier/repl.sh` and `TutorialSession.benchmark(...)` for rapid edit/run
cycles. Edits reload the student's function in a fresh module; unchanged code, loaded
libraries, and compiled kernels are retained between runs. Start with the small default
correctness checkpoint, then increase matrix/tile size for performance. Show
the first compilation delay explicitly, then demonstrate the warm feedback
loop. Call `TutorialSession.plot()` after each configuration so everyone sees
the terminal throughput chart change. The reference solution is an environment
check; the supplied student template must initially fail.

Use `harness/plot.py` on the laptop to export PNG/SVG performance charts from
copied CSV results. `frontier/fetch-results.sh` uses a second persistent SSH
transport so transfers work while the interactive terminal remains connected.
Keep a browser open to the SVG for the audience, or put the
PNG into slides. Use `TutorialSession.trace(...)` and open the copied JSON in
Perfetto when discussing task overlap and data movement. Trace timings include
logging overhead and describe host-side task events; show separate untraced
benchmark numbers. Each participant writes their own result file by default;
combine participant CSVs only after the session rather than letting concurrent
users append to one shared file.

For the same-node MPI demo, use `SC26_NODES=1 bash frontier/mpi-repl.sh 4` in a second
tmux pane. It warms up once, then broadcasts harness option lines from rank 0
so all ranks rerun the edited implementation without restarting Julia. Use the
four-rank/one-node matrix-product demo before the Cholesky run. The default REPL step
holds all eight GPUs on one node; exit it before MPI steps that need that node.
Alternatively, `SC26_GPUS_PER_TASK=2 bash frontier/repl.sh` limits the warm REPL
to two GPUs, leaving other devices for separate steps under the same allocation.
Use the output-tile reports and distinct-device checks to establish GPU
participation. Keep validated logs,
charts, and a task trace as presentation fallbacks.

## Demo presentation and local console validation (October 22, 2026)

The highlighting follow-up uses Highlights.jl 0.6.3 with the Tree-sitter Julia
grammar and a Dracula theme, including readable gray comments on dark panels.
Both terminal and web output retain the same 24-bit token colors. Preparation
resolves new dependencies before instantiating existing manifests and caches
the highlighter's theme artifacts on the login node. The two-GPU matrix demo
passed in portal job 45136 on `odo07` with the new dependencies and no compute
node downloads. Source events contained true-color escapes; browser rendering
preserved them. The revised checks passed 21 Julia checks, eight Python tests,
and the JavaScript ANSI conversion tests (including real highlighter output).

The console now listens on `0.0.0.0:8767` by default for Tailscale access.
IP addresses and detected MagicDNS hostnames pass its request checks; unknown
DNS hostnames remain rejected. `--host 127.0.0.1` selects loopback-only use.
HTTP access through the host's Tailscale address was verified.

All seven demos now display marked algorithm excerpts from their actual source,
enable Dagger compute/movement logging, print a UnicodePlots Gantt, and save
portable JSON timelines under `results/demo-timelines/`. MPI reports combine
both ranks' logs; processor labels retain rank identity. Timings are host task
events, including compilation, rather than GPU kernel profiles. The source
excerpts remain complete Julia syntax and omit presentation plumbing.

The optional host app (`python3 host/app.py`) serves a local webpage on port
8767 and uses a dedicated SSH transport. A login-side coordinator prepares
packages before requesting a one-node Slurm holding job; persistent Julia
steps receive demo or evaluator commands. Single-process work exposes four
GPUs; MPI exposes two per rank on the same node. Switching layouts restarts
Julia. Release, EOF and local shutdown clean up only the console's own job.
The app uploads into its own marked scratch directory, leaving normal
checkouts and exercise edits separate. See `host/README.md` for setup and the
supported evaluator interface.

Live checks on `odo04` passed all seven demos, single-process two- and four-GPU
matrix products, and the two-rank/four-distinct-GPU MPI product. The CPU-first
MPI workflow also passed: GPU support is loaded before MPI builds its processor
inventory. Repeated runs retained the Julia worker, and both single-process
and MPI evaluator variables persisted across commands. Cancelling a sleeping
MPI expression stopped the step while retaining its allocation; the subsequent
single-process evaluator and four-GPU demo passed. Reconnecting obtained a
fresh allocation and reran CPU/MPI/GPU checks without world-age warnings.
Validation used portal jobs 45133 and 45134. JSON traces remain in
`/gpfs/wolf2/olcf/trn047/scratch/jpsamaroo/sc26-dagger/portal/results/demo-timelines/`;
the local validation transcript is `results/validation/portal-live.json`.

The local suite passed 19 Julia checks and seven Python tests covering source
extraction, event durations, lane placement, the UnicodePlots backend, command
serialization, cancellation ownership, HTTP boundaries and stream backpressure.
Browser checks verified highlighted source, timeline tooltips, transfer toggles,
zoom and history selection without JavaScript errors or horizontal page overflow.
These checks validate the console and demos, not class-wide allocation timing
or the complete Cholesky smoke suite, which remain pre-event checks.

## Interactive startup validation (October 21, 2026)

Validated the scratch checkout on Odo with Julia 1.13.1, Dagger 0.22.7, and
the pinned AMDGPU revision. `allocate.sh` completed package preparation on
`login1` before `salloc` opened the allocation shell on `odo03`. Calling
`prepare.sh` from that compute shell stopped with the login-node instruction.

Within the same one-node allocation, `repl.sh` opened a Julia prompt with
`JULIA_PKG_OFFLINE=true`. The tasks demo and reference CPU/GPU checks passed
at size 1024 with 256-wide tiles. After exiting Julia, `mpi-repl.sh 4` opened
its persistent prompt with one visible GPU per rank. Its initial strict GPU
check and subsequent strict CPU/GPU checks all passed. The first MPI GPU
warm-up reported 59.4 seconds; a repeated GPU warm-up reported below 0.1 seconds.
Package compilation occurred on the compute node, without package preparation
or a package-server error there. These were targeted startup and interactive
checks; the complete revised smoke suite remains a pre-event check.

The matrix-product demo also passed on `odo03` with two and four GPUs in one
Julia process, and with four MPI ranks using one GPU each on that node. Each
run assembled a single `C = A * B`, checked it against the dense reference, and
verified distinct physical devices using hostname plus PCI address returned
from the actual matrix-multiplication tasks. Logs are under
`results/validation/matmul-{2gpu,4gpu,mpi}.log` in the Odo scratch checkout.

Cross-host validation of the same matrix-product implementation also passed
across `odo06` and `odo07`: the CPU demo
reported two hosts and two Julia processes; `multinode.sh gpu` reported two
GPU devices on each host, four distinct devices total, and two Julia processes.
Both compared their assembled matrix product against the dense reference.
The shared implementation also passed the original one-node/two-GPU demo.
The GPU launch uses `per_task:2` binding: the earlier `closest` binding exposed
only one device despite requesting two per rank. Logs are
`results/validation/multinode-{cpu,gpu}.log` on Odo. The workshop launcher now
places both ranks on one node; the same Julia algorithm supports both layouts.

The current workshop layout passed on `odo03`: two CPU ranks reported zero
GPUs, one node, and two Julia processes; two GPU ranks reported four distinct
physical devices, one node, and two Julia processes. Both verified `C ≈ A * B`.
Logs are `results/validation/mpi-one-node-{cpu,gpu}.log` in the scratch checkout.
The GPU run initially faulted with same-node ROCm IPC enabled and passed after
setting `DAGGER_IPC=false`; `frontier/env.sh` now defaults to host staging for
both local and cross-host MPI transfers. The workshop wrapper also passed
with a stale `SC26_NODES=2` in its caller, confirming it still selects one node.
These targeted demo checks do not replace the complete pre-event smoke suite.

## Validation notes (October 14, 2026)

These historical measurements include the earlier two-node layout. The current
course uses one node per person; rerun the revised smoke suite on Odo to validate
the four-rank/one-node GPU path before the event.

The initial reference suite passed all 16 compute checks on `odo07`/`odo08`:
CPU, GPU, all-eight-GPU, mixed CPU/GPU, two MPI ranks on one node, two ranks on
two nodes, and four GPU ranks across two nodes. The incomplete template produced
the intended diagnostic. An intentionally incorrect solution marking all tiles
`In` failed the strict MPI residual check. A real batch job also completed with
exit code zero. Two independent Slurm steps overlapped for five seconds on
`odo08`, with distinct `ROCR_VISIBLE_DEVICES=0` and `1`, confirming resource
sharing within one owner's allocation.

The larger 16384-by-16384 run passed residual checks but exposed an AMDGPU 2.8.0
handle-cache finalizer bug. `Project.toml` therefore pins AMDGPU to upstream
revision `482cd054172541a0ef5e4626f6c3505fc3ae7972`, including the destructor and
context-ownership fixes. Dagger stays on master (`034f72a`, version 0.22.7).
The corrected environment passed the full suite plus the incomplete-template
check, CPU/GPU built-in and vendor baselines, and the larger 16384-size run;
no finalizer errors remained. The four-rank/two-node interactive MPI prompt
passed CPU, GPU, and mixed targets, plotting, missing-file recovery, and forced
reload. Retaining unchanged implementation modules reduced repeat strict GPU
warm-up to below the displayed 0.1-second precision; timed 1024-size trials were
about 0.026–0.031 seconds. Edited files reload automatically, and source snapshots
are broadcast from rank 0 so ranks compile the same version. Use `--reload` for
changes to included helper files.

Cold MPI GPU warm-up took approximately 36–54 seconds in the initial runs;
warm timed 1024-size trials took roughly 0.02–0.30 seconds. Prelaunch MPI demos
and make compilation visible to the audience. The launcher gives each MPI rank
seven physical cores and one visible GPU. Use 56 threads for a whole-node
single-process scaling run if desired (`SC26_THREADS=56`). The measured one-GPU
and eight-GPU results at sizes up to 16384 did not show an eight-GPU speedup;
show placement and correctness separately from performance, and use the plots
to discuss tile size, task overhead, communication, and the Cholesky critical
path. These samples are rehearsal measurements, not peak-performance claims.
