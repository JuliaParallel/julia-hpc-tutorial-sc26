# Exercise: parallel Cholesky with Dagger (30 minutes)

You'll turn a serial, tile-by-tile Cholesky factorization into a parallel
Dagger program. Then you'll run the **same function, unchanged** on CPU
threads, on an AMD GPU, and across MPI ranks.

Edit `exercise/my_cholesky.jl` following the steps below. Allow about
12 minutes for the CPU implementation, 8 minutes for the GPU run,
6 minutes for MPI, and 4 minutes to plot and discuss results.

Run all commands from `02-Dagger/`. For a laptop or workstation, use
`julia --project -t 4 harness/run.jl` in place of `bash frontier/run.sh 1`;
see the [quickstart](../README.md) for the local MPI launch command.

## The algorithm in one picture

For each step `k`, working on a grid of tiles `T[i,j]` (lower triangle only):

```
 k=1:  [P]              P = POTRF  factor the diagonal tile          T[k,k]
       [S][U]           S = TRSM   solve the tiles below it          T[m,k] (reads T[k,k])
       [S][G][U]        U = SYRK   update diagonal tiles             T[n,n] (reads T[n,k])
       [S][G][G][U]     G = GEMM   update off-diagonal tiles         T[m,n] (reads T[m,k], T[n,k])
```

All the `S` tasks can run at once, and so can all the `U`/`G` tasks. Step
`k+1` can even start before step `k` finishes. You don't have to work out
any of that; Dagger does, from your `In`/`InOut` annotations.

## Steps

**0. Get onto a compute node** (do this during the talk):

```bash
source frontier/env.sh
bash frontier/allocate.sh
```

**1. Make it parallel (CPU).** In `exercise/my_cholesky.jl`:

1. Wrap the loops in `Dagger.spawn_datadeps() do ... end`.
2. Put `Dagger.@spawn` before each of the 4 BLAS/LAPACK calls.
3. Wrap each tile argument in `In(...)` (only read) or `InOut(...)` (read and written).

Keep the existing algorithm and loop structure, and return `T`. Each
`T[i,j]` is a Dagger handle to a tile. Spawn each BLAS/LAPACK function
directly so Dagger can select its GPU equivalent. Avoid `fetch`, `wait`,
`@sync`, or `Threads.@spawn` inside the Datadeps region, and keep GPU- and
MPI-specific code out of `my_cholesky!`.

```bash
bash frontier/run.sh 1 --impl=student
```

Done when every line says `PASS`.

**2. Same code, on the GPU.** No edits:

This step requires a working AMD GPU environment. If the GPU path is
unavailable, use the instructor demonstration and continue to MPI.

```bash
bash frontier/run.sh 1 --impl=student --target=gpu
```

**3. Same code, on MPI.** No edits:

```bash
bash frontier/run.sh 2 --impl=student --mpi --strict
```

`--strict` makes Dagger place tasks away from their data, so a missing
`InOut` loses its writes and fails the check. It can pass without
`--strict` by luck.

**4. See how you did:**

```bash
julia --project harness/plot.jl --label=$USER
```

## Stretch goals

- **Tune the tile size:** `--bs=512`, `--bs=1024`, `--bs=4096` on the GPU. Why is there a sweet spot?
- **Compare:** `--impl=builtin` (Dagger's own Cholesky) and `--impl=vendor`
  (one LAPACK or rocSOLVER call on the whole matrix).
- **CPU and GPU together:** `--target=mixed`. Is it faster? Why not?
- Look at `solution/my_cholesky.jl` only once you're done.

## When something goes wrong

| You see | Likely cause |
|---|---|
| `MethodError: no method matching potrf!(::Char, ::DTask)` | A call is missing `Dagger.@spawn`, or it's outside `spawn_datadeps`. Outside a task, tiles are Dagger handles, not matrices. |
| `FAIL: warm-up residual ...` | A tile that's written is marked `In` (or not marked at all). Written tiles need `InOut`. |
| GPU run errors inside your own helper function | Spawn `BLAS.gemm!(...)` itself, not a function that calls it. Dagger swaps BLAS for rocBLAS only on the spawned function. |
| MPI run hangs | Code differs between ranks (e.g. `if rank == 0`). Every rank must run identical Dagger code. |
