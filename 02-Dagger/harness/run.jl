# Eval + benchmark harness for the Dagger Cholesky exercise.
#
# Runs a Cholesky implementation, checks the answer, times it, and appends
# the results to results/results.csv (see harness/plot.jl to plot them).
#
#   julia --project -t 4 harness/run.jl [options]
#
# Options:
#   --impl=student|solution|builtin|vendor   what to run (default: student)
#       student  -> exercise/my_cholesky.jl
#       solution -> solution/my_cholesky.jl
#       builtin  -> Dagger's own DArray Cholesky
#       vendor   -> one LAPACK (CPU) or rocSOLVER (GPU) call on the whole matrix
#   --file=path/to/file.jl    load my_cholesky! from here instead
#   --target=cpu|gpu|mixed    where Dagger may run tasks (default: cpu)
#   --sizes=2048,4096         matrix sizes N (default depends on target)
#   --bs=512                  tile size (default depends on target)
#   --trials=3                timed runs per size
#   --label=NAME              tag stored with your results (default: $USER)
#   --mpi                     run under MPI (launch with srun/mpiexec)
#   --strict                  spread tasks away from their data, to catch missing
#                             InOut annotations (use with --mpi)
#
# Examples:
#   julia --project -t 4 harness/run.jl --impl=student --target=cpu
#   julia --project -t 4 harness/run.jl --impl=student --target=gpu
#   srun -n 2 julia --project -t 3 harness/run.jl --impl=student --mpi

const USE_MPI = "--mpi" in ARGS
const USE_GPU = any(a -> a in ("--target=gpu", "--target=mixed"), ARGS)

if USE_MPI
    using MPI
end
if USE_GPU
    using AMDGPU
end
using Dagger

USE_MPI && Dagger.accelerate!(:mpi)

include(joinpath(@__DIR__, "common.jl"))

function main(args)
    o = parse_options(args)
    # Each Dagger task runs one BLAS call; don't let BLAS spawn threads too.
    BLAS.set_num_threads(1)

    say("== Dagger Cholesky harness ==")
    say("impl=$(o.impl)  target=$(o.target)  ranks=$(nranks())  threads/rank=$(Threads.nthreads())  bs=$(o.bs)", o.strict ? "  STRICT" : "")
    USE_GPU && say("GPU: ", AMDGPU.device(), "  functional=", AMDGPU.functional())

    run! = make_runner(o)

    # Warm-up on a small problem: compiles everything, and catches wrong
    # answers in seconds instead of after the big runs.
    nw = 4 * o.bs
    Aw = make_spd(nw; seed=o.seed)
    say("warm-up (n=$nw, includes compilation)...")
    tw, Lw = try
        run!(Aw)
    catch err
        say("FAIL: your code threw an error during warm-up:\n")
        if myrank() == 0
            # SC26_DEBUG=1 prints the full stack trace (for instructors)
            get(ENV, "SC26_DEBUG", "0") == "1" ? showerror(stdout, err, catch_backtrace()) :
                                                 showerror(stdout, err)
        end
        say("\n\nHint: outside a Dagger task, tiles are Dagger handles, not arrays --")
        say("did every BLAS/LAPACK call get a `Dagger.@spawn` inside `spawn_datadeps`?")
        return 1
    end
    rw = scaled_residual(Aw, Lw)
    if !(rw < RESID_TOL)
        say(@sprintf("FAIL: warm-up residual %.3g (needs < %g). Check your In/InOut annotations!", rw, RESID_TOL))
        return 1
    end
    say(@sprintf("  ok (%.1f s, residual %.3g)", tw, rw))

    failed = false
    for n in o.sizes
        A = make_spd(n; seed=o.seed + n)
        for trial in 1:o.trials
            t, L = run!(A)
            r = scaled_residual(A, L)
            ok = r < RESID_TOL
            failed |= !ok
            record!(o, n, trial, t, r)
            say(@sprintf("  n=%6d  trial %d: %8.3f s  %9.1f GFLOP/s  resid=%.3g  %s",
                         n, trial, t, cholesky_gflops(n, t), r, ok ? "PASS" : "FAIL"))
        end
    end
    say("results appended to ", o.out, "  (plot: julia --project harness/plot.jl)")
    return failed ? 1 : 0
end

code = main(ARGS)
USE_MPI && MPI.Barrier(MPI.COMM_WORLD)
exit(code)
