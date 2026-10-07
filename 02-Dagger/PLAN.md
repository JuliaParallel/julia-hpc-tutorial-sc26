# SC26 Julia tutorial: Dagger.jl for HPC (1 hour)

This session follows the introduction to Julia and precedes vendor-neutral
parallel programming. It includes 30 minutes of teaching and demos and
30 minutes of hands-on work.

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
| 20:00–25:00 | MPI | `Dagger.accelerate!(:mpi)` and SPMD: every rank runs the same Dagger code and makes the same decisions. | `demos/04_mpi.jl` |
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
8. MPI: SPMD and uniform decisions across ranks (`demos/04_mpi.jl`).
9. Tiled Cholesky: four kernels and a dependency graph for a 4×4 tile grid.
10. Exercise: edit the template, check correctness, then change execution targets.
11. Results: tile-size effects and comparison with built-in and vendor implementations.
12. Discussion: when task parallelism helps and how data movement affects performance.

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

- Obtain Odo compute allocations before the exercise, using `frontier/allocate.sh`.
- Confirm the Odo configuration placeholders in `frontier/*.sh` and build the shared depot with `frontier/setup.sh`.
- Validate the CPU, GPU, and MPI commands on the tutorial hardware. GPU runs have shown intermittent failures in local ROCm 7.2 testing; Odo validation is still required.
- Run each demo before the session and keep a warm Julia REPL for demos 1–3.
- Launch MPI in a separate terminal. Prepare recorded GPU and MPI demonstrations as fallbacks.
- If GPU execution is unavailable, use the instructor demonstration for that checkpoint and continue with CPU and MPI work.

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
