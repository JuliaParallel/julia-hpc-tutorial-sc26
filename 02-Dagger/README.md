# SC26 Julia tutorial: Dagger.jl for HPC

Hands-on materials for the **1-hour Dagger segment**: 30 minutes of teaching
and demos, followed by 30 minutes of hands-on work. Students parallelize a tiled
Cholesky factorization with Dagger's Datadeps, then run the same function on
CPU threads, an AMD GPU, and MPI ranks. The session guide is in
[`PLAN.md`](PLAN.md); the student handout is [`exercise/README.md`](exercise/README.md).

## Quickstart on a laptop or workstation

Run these commands from `02-Dagger/`:

```bash
julia --project -e 'using Pkg; Pkg.instantiate()'

julia --project -t 4 harness/run.jl --impl=solution               # CPU threads
julia --project -t 4 harness/run.jl --impl=solution --target=gpu  # AMD GPU
julia --project -e 'using MPI; run(`$(MPI.mpiexec()) -n 2 $(Base.julia_cmd()) --project -t 2 harness/run.jl --impl=solution --mpi --strict`)'
julia --project harness/plot.jl
```

Use `--impl=student` to check `exercise/my_cholesky.jl`, or `--file=x.jl` to
check any file. All options are listed at the top of `harness/run.jl`.

## On Odo (Frontier training system)

```bash
source frontier/env.sh
bash frontier/allocate.sh                                # get a compute allocation first
bash frontier/run.sh 1 --impl=student                    # CPU
bash frontier/run.sh 1 --impl=student --target=gpu       # GPU
bash frontier/run.sh 2 --impl=student --mpi --strict     # MPI
```

Instructors: build the shared, pre-compiled package depot once with
`frontier/setup.sh`. The Odo scripts contain `TODO(confirm)` placeholders
and haven't been run yet.

## GPU validation

GPU runs have shown intermittent failures in local ROCm 7.2 testing.
The GPU demos and exercise must be validated on Odo's MI250X before the
tutorial. If the GPU path is unavailable, complete the CPU and MPI steps
and use an instructor demonstration for the GPU step.

The harness saves benchmark results in `results/`, which is excluded from
Git because the CSV files include user labels and hostnames.
