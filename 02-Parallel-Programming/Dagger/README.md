# SC26 Julia tutorial: Dagger.jl for HPC

Hands-on materials for the **1-hour Dagger segment**: 30 minutes of teaching
and demos, followed by 30 minutes of hands-on work. Students parallelize a tiled
Cholesky factorization with Dagger's Datadeps, then run the same function on
CPU threads, an AMD GPU, and MPI ranks. The session guide is in
[`PLAN.md`](PLAN.md); the student handout is [`exercise/README.md`](exercise/README.md).

## Slides (Quarto)

The [Dagger slide deck](presentations/dagger/index.qmd) builds on the preceding
parallel, portable GPU programming session. It combines visual explanations,
live-demo cues, timed exercise checkpoints, recorded Odo measurements, and
speaker notes. Optional launch, solution, resource, and environment references
follow the closing slide.

```bash
cd presentations/dagger
quarto render index.qmd
quarto preview index.qmd
```

Present `index.html` in a browser. Arrow keys advance, `F` enters fullscreen,
`Esc` opens the overview, and `S` opens speaker notes when served over HTTP.
Use the browser's print dialog on `index.html?print-pdf` to save a PDF.
The rendered HTML embeds its assets and works offline; it does not execute
Julia or need access to Odo. Generated HTML/PDF files are ignored by Git.

Edit `theme.scss` for styling. `python3 build_figures.py` regenerates the SVG
diagrams and charts using Python's standard library. The adjacent `data/`
directory contains the October 14 rehearsal extracts, with usernames and
hostnames omitted. These figures describe that tested environment, rather
than promising a speedup or current cluster availability.

## Quickstart on a laptop or workstation

Run these commands from `02-Parallel-Programming/Dagger/`:

```bash
julia --project -e 'using Pkg; Pkg.instantiate()'

julia --project -t 4 harness/run.jl --impl=solution               # CPU threads
julia --project -t 4 harness/run.jl --impl=solution --target=gpu  # AMD GPU
julia --project -e 'using MPI; run(`$(MPI.mpiexec()) -n 2 $(Base.julia_cmd()) --project -t 2 harness/run.jl --impl=solution --mpi --strict`)'
julia --project harness/plot.jl
```

Use `--impl=student` to check `exercise/my_cholesky.jl`, or `--file=x.jl` to
check any file. All options are listed at the top of `harness/run.jl`.

### Independent access for 25 attendees and one presenter

Reserve **26 healthy compute nodes**: one per attendee and one for the presenter.
Everyone keeps their own account, code, results, and allocation. All tutorial
MPI examples use multiple ranks on one node, so the existing exclusive-node
policy supports concurrent CPU, multi-GPU, and MPI work without node sharing.
The recorded 29 healthy nodes leave three spare if otherwise available.

See the [resource plan](PLAN.md#resource-plan-for-25-attendees-and-one-presenter)
for reservation requirements and a fallback if fewer than 26 nodes are available.
That fallback uses two windows of 13 and 12 independent sessions, plus one
presenter node (14 nodes at peak), with local CPU work outside each window.
Same-node MPI exercises process communication and placement; it does not measure
communication between physical hosts.

## On Odo (Frontier training system)

The environment tracks Dagger's `master` through `[sources]` in `Project.toml`.
A fresh `Pkg.instantiate()` resolves that branch; an existing environment stays
at its resolved commit until `Pkg.update("Dagger")`. The validation baseline is
Dagger 0.22.7, master commit `034f72a85971b9ada81fa09ab6ca037e7f0fe72d`.
Update and repeat validation before freezing the course environment.
AMDGPU is pinned to `482cd054172541a0ef5e4626f6c3505fc3ae7972`, which includes
upstream fixes for library-handle cleanup and GPU context ownership. The registry
2.8.0 build produced `destroy_handle!` finalizer errors during the larger run.
This revision fixes those errors while retaining compatibility with Dagger's
KernelAbstractions 0.9 dependency; AMDGPU's newer main branch requires 0.10.

Use the login node for downloads and compute nodes for precompilation and work:

```bash
export SC26_NODES=1                              # course layout for every launch
bash frontier/allocate.sh                        # login node: prepare packages, then allocate
source frontier/env.sh
srun -N1 -n1 -c7 --threads-per-core=1 --gpus-per-task=8 --gpu-bind=none bash frontier/setup.sh
bash frontier/run.sh 1 --impl=solution            # check the reference first
bash frontier/run.sh 1 --impl=student             # CPU checkpoint
bash frontier/run.sh 1 --impl=student --target=gpu # GPU checkpoint
bash frontier/run.sh 2 --impl=student --mpi --strict
julia --project harness/plot.jl
```

`frontier/env.sh` selects Julia 1.13.1, ROCm 6.4.2, GNU, and Cray MPICH 8.1.31
on Odo. The cache defaults to your project scratch directory on wolf2, not
Frontier's Orion filesystem. Set `SC26_ACCOUNT`, `SC26_DEPOT`, and
`SC26_SHARED_DEPOT` as appropriate. `prepare.sh` must run on a login node:
compute nodes cannot reach the Julia package server. `allocate.sh` runs it
automatically before calling `salloc`. For an allocation obtained another way,
run `bash frontier/prepare.sh` on the login node first. On Odo, `salloc` opens
the allocation shell on a compute node, so downloads must finish before that
transition. Each user needs a writable first depot; instructors can prepare a
readable shared depot using `SC26_DEPOT`.
Keep MPI preferences in each project copy (`LocalPreferences.toml`).

Every Slurm job currently occupies whole nodes, even when requesting one GPU.
Slurm exposes the eight MI250X GCDs as eight devices per node. The launcher
uses seven physical CPU cores per rank, binds one GPU per MPI rank, and exposes
all eight GPUs to a single process. Override `SC26_THREADS` or
`SC26_GPUS_PER_TASK` for a smaller step. `SC26_NODES` selects the number of nodes
used by a step, defaulting to one. MPI examples use host-staged GPU transfers.

### Multi-GPU and multiple MPI ranks on one node

All commands below use the same one-node allocation. Explicitly select the
course layout, including when reusing a shell from an earlier session:

```bash
export SC26_NODES=1
SC26_GPUS_PER_TASK=2 bash frontier/launch.sh 1 demos/05_multigpu.jl
SC26_GPUS_PER_TASK=2 bash frontier/run.sh 1 --impl=solution --target=gpu --gpus=all
bash frontier/launch.sh 2 demos/04_mpi.jl
bash frontier/run.sh 2 --impl=solution --mpi --strict
bash frontier/launch.sh 4 demos/05_multigpu.jl --mpi
bash frontier/run.sh 4 --impl=solution --mpi --strict --target=gpu
```

The four-rank launch uses four GPUs on one node, one per rank. Each rank sees
its own GPU as device 1, so the algorithm and GPU scope stay unchanged. The
topology demo runs a tile on every selected device and prints one shared
hostname, distinct ranks, and device bindings.

For batch execution, after preparing the environment:

```bash
sbatch -A trn047 frontier/batch.sh 1 --impl=solution --target=gpu --gpus=all
sbatch -A trn047 frontier/batch.sh 4 --impl=solution --mpi --strict --target=gpu
```

Run the smoke suite with `bash frontier/validate.sh` inside a one-node allocation.
It checks two-rank CPU/GPU execution and four-rank GPU placement and correctness
on the same node. Logs are saved under `results/validation/`; correctness
assertions and harness failures produce nonzero exits. The student template is
intentionally incomplete and should fail until annotated; use the solution for
environment validation.

### Fast feedback and plots over SSH

Keep Julia alive while editing so loaded packages, GPU initialization, and compiled
functions/kernels are retained. Each call runs a complete new check; edited code
reloads automatically. The Slurm allocation separately holds the node until its
shell exits, even after Julia exits:

```bash
bash frontier/repl.sh
```

Both REPL launchers use the environment prepared by `allocate.sh` before it
requests compute resources. That preparation instantiates packages in the
tutorial depot, downloads required lazy artifacts, and configures Cray MPI on
the login node, with automatic precompilation disabled. The REPL launchers do
not run package preparation; they set Julia's package manager to offline mode.
Julia compiles packages on the compute node as they load. The first allocation
can take longer while dependencies are installed.

In that REPL:

```julia
TutorialSession.benchmark(impl="solution")
TutorialSession.benchmark(impl="student") # reloads edits; retains compiled code otherwise
TutorialSession.benchmark(impl="solution", target="gpu", sizes=[2048], bs=512)
TutorialSession.benchmark(impl="solution", target="gpu", gpus="all", sizes=[4096], bs=512)
TutorialSession.plot()                   # chart in the SSH terminal
```

Defaults favor small correctness checkpoints and one timed trial. Opt into
`--sizes=4096,8192,16384 --bs=1024 --trials=3` after correctness is established.
Tiny problems demonstrate correctness and placement; they do not measure peak
GPU performance. Compilation and warm-up are reported separately from trials.
Exit the REPL before launching a step that needs its reserved CPUs/GPUs.

For repeated MPI runs, keep four Julia processes and their MPI communicator alive
on the same node too:

```bash
SC26_NODES=1 bash frontier/mpi-repl.sh 4
```

After its initial warm-up, enter harness options at the `mpi>` prompt:

```text
--impl=solution --target=gpu --sizes=2048 --bs=512 --strict
--impl=student --target=gpu --sizes=2048 --bs=512 --strict
plot
quit
```

Rank 0 broadcasts each command to all ranks. Edited implementation files reload automatically from the shared filesystem;
unchanged functions and CPU/GPU/MPI code stay compiled. Use `--reload` when
you changed an included helper file rather than the implementation itself.
This prompt accepts harness options plus `plot`, `help`, and `quit`. For a CPU
warm-up instead, set `SC26_MPI_WARM_TARGET=cpu`.

For a presentation-quality chart, copy the results CSV to your laptop, then:

```bash
bash frontier/fetch-results.sh                    # copies CSV and exports both charts
# Or plot an already-downloaded CSV:
python3 harness/plot.py results/odo-results.csv --output results/odo-performance
```

This produces PNG and SVG charts with throughput and elapsed time. It needs
Matplotlib only on the laptop (`python3 -m pip install matplotlib`). Terminal
plots require no display server, Jupyter, browser server, or SSH forwarding.

For a browser task timeline, first warm the chosen configuration, then:

```julia
TutorialSession.trace(impl="solution", target="gpu", sizes=[2048], bs=512)
```

Copy `results/timeline.json` to the laptop and open it in
[Perfetto](https://ui.perfetto.dev/). The lanes retain Dagger CPU/GPU processor
names. These are host-side task and data-movement events, not hardware kernel
profiling; tracing adds overhead, so use untraced trials for performance.

See [`PLAN.md`](PLAN.md) for resource planning and presenter logistics.

The harness saves benchmark results in `results/`, which is excluded from
Git because the CSV files include user labels and hostnames.
