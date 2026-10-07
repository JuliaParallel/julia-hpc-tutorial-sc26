# Shared pieces of the eval/benchmark harness: options, test matrices,
# tiling, validation, timing, and CSV output. Included by run.jl.
#
# `USE_MPI` and `USE_GPU` must be defined (and MPI/AMDGPU loaded) before
# this file is included.

using Dagger
using LinearAlgebra
import LinearAlgebra: BLAS, LAPACK
using Random, Printf, Dates

const ROOT = dirname(@__DIR__)

# ---------------------------------------------------------------------------
# Options

Base.@kwdef mutable struct Options
    impl::String = "student"   # student | solution | builtin | vendor
    target::String = "cpu"     # cpu | gpu | mixed
    sizes::Vector{Int} = Int[] # empty -> default for target
    bs::Int = 0                # tile size; 0 -> default for target
    trials::Int = 3
    seed::Int = 1234
    file::String = ""          # overrides the file loaded for student/solution
    strict::Bool = false       # place tasks away from their data (see `strictly`)
    label::String = get(ENV, "USER", "anon")
    out::String = joinpath(ROOT, "results", "results.csv")
end

const DEFAULT_SIZES = Dict("cpu"   => [1024, 2048, 4096],
                           "gpu"   => [4096, 8192, 16384],
                           "mixed" => [4096, 8192])
const DEFAULT_BS    = Dict("cpu" => 256, "gpu" => 2048, "mixed" => 1024)

function parse_options(args)
    o = Options()
    for a in args
        a == "--mpi" && continue            # handled before include
        a == "--strict" && (o.strict = true; continue)
        startswith(a, "--") || error("Unrecognized argument: $a")
        key, val = occursin('=', a) ? split(a[3:end], '='; limit=2) : (a[3:end], "")
        if key == "impl";        o.impl = val
        elseif key == "target";  o.target = val
        elseif key == "sizes";   o.sizes = parse.(Int, split(val, ','))
        elseif key == "bs";      o.bs = parse(Int, val)
        elseif key == "trials";  o.trials = parse(Int, val)
        elseif key == "seed";    o.seed = parse(Int, val)
        elseif key == "file";    o.file = val
        elseif key == "label";   o.label = val
        elseif key == "out";     o.out = val
        else error("Unknown option --$key")
        end
    end
    o.impl in ("student", "solution", "builtin", "vendor") ||
        error("--impl must be student, solution, builtin, or vendor")
    o.target in ("cpu", "gpu", "mixed") || error("--target must be cpu, gpu, or mixed")
    o.target != "cpu" && !USE_GPU && error("--target=$(o.target) needs AMDGPU")
    o.impl == "vendor" && USE_MPI && error("--impl=vendor is a single-process baseline; drop --mpi")
    isempty(o.sizes) && (o.sizes = DEFAULT_SIZES[o.target])
    o.bs == 0 && (o.bs = DEFAULT_BS[o.target])
    for n in o.sizes
        n % o.bs == 0 || error("size $n is not a multiple of tile size $(o.bs)")
    end
    return o
end

# ---------------------------------------------------------------------------
# MPI helpers (no-ops without --mpi)

myrank() = USE_MPI ? MPI.Comm_rank(MPI.COMM_WORLD) : 0
nranks() = USE_MPI ? MPI.Comm_size(MPI.COMM_WORLD) : 1
barrier() = USE_MPI ? MPI.Barrier(MPI.COMM_WORLD) : nothing
allreduce_max(x) = USE_MPI ? MPI.Allreduce(x, max, MPI.COMM_WORLD) : x
say(args...) = myrank() == 0 && println(args...)

# ---------------------------------------------------------------------------
# Where tasks may run

function target_scope(target)
    cpu = USE_MPI ? Dagger.scope(mpi_ranks=:) : Dagger.scope(worker=1)
    target == "cpu" && return cpu
    gpu = Dagger.scope(rocm_gpu=1)   # under MPI: GPU 1 of every rank
    target == "gpu" && return gpu
    return Dagger.UnionScope(cpu, gpu)
end

with_target(f, target) = Dagger.with_options(f; scope=target_scope(target))

gpu_sync(target) = target == "cpu" || Dagger.gpu_synchronize(:ROC)

# ---------------------------------------------------------------------------
# Test problem

"""
Symmetric, strictly diagonally dominant (hence SPD) test matrix. Seeded, so
every MPI rank builds the identical matrix -- a Dagger+MPI requirement.
"""
function make_spd(n; seed)
    A = rand(Xoshiro(seed), n, n)
    A = A + A'
    A[diagind(A)] .+= 2n
    return A
end

"""
Tile `A` into a bs x bs DArray whose tiles live where the target computes
(so GPU timings don't include host<->device copies).

GPU runs must be validated on the tutorial hardware; see ../README.md.
"""
to_darray(A, bs, target) = with_target(() -> DArray(A, Blocks(bs, bs)), target)

"""
HPL-style scaled residual, ||A x - L (L' x)||_inf / (||A||_inf ||x||_inf n eps).
Costs O(n^2) rather than the O(n^3) of forming L*L'. Passes if < 16.
"""
function scaled_residual(A, Lfull)
    n = size(A, 1)
    L = LowerTriangular(Lfull)
    x = rand(Xoshiro(7), n)
    r = A * x - L * (L' * x)
    return norm(r, Inf) / (norm(A, Inf) * norm(x, Inf) * n * eps())
end
const RESID_TOL = 16.0

cholesky_gflops(n, t) = (n^3 / 3) / t / 1e9

# ---------------------------------------------------------------------------
# Implementations under test. Each returns (seconds, lower-triangular factor).

"Load a user file into a fresh module so student/solution don't collide."
function load_impl(path)
    m = Module(Symbol(splitext(basename(path))[1]))
    Core.eval(m, :(include(x) = Base.include($m, x)))
    Base.include(m, path)
    return m
end

"""
Datadeps trusts your In/InOut annotations. Normally Dagger runs each task where
its data already lives, so a missing `InOut` can still give right answers by
luck. `--strict` turns off data-affinity placement, so tasks get spread across
ranks/devices and anything written without `InOut` is lost -> wrong answer.
Most useful under MPI (with plain threads everything is in-place anyway).
"""
strictly(f, strict) = strict ? Base.ScopedValues.with(f, Dagger.DATADEPS_HIERARCHICAL => false) : f()

function run_darray(f!, A, o)
    DA = to_darray(A, o.bs, o.target)
    barrier()
    t0 = time_ns()
    with_target(() -> strictly(() -> f!(DA), o.strict), o.target)
    gpu_sync(o.target)
    barrier()
    t = allreduce_max((time_ns() - t0) / 1e9)
    return t, gather(DA)
end

"""
Assemble the full matrix on the host. Works around Dagger 0.22.5's `collect`
failing on a DArray with tiles in both CPU and GPU memory (`--target=mixed`):
it `cat`s a Matrix with a ROCArray, which scalar-indexes the GPU array.
Under MPI we need `collect` (only it gathers across ranks), so mixed+MPI
still hits this.
"""
function gather(DA)
    USE_MPI && return collect(DA)
    tiles = map(c -> Array(fetch(c)), DA.chunks)
    return reduce(vcat, [reduce(hcat, tiles[i, :]) for i in axes(tiles, 1)])
end

function run_vendor(A, o)
    if o.target == "cpu"
        B = copy(A)
        BLAS.set_num_threads(Threads.nthreads())
        t = @elapsed LAPACK.potrf!('L', B)
        BLAS.set_num_threads(1)
        return t, B
    else
        dA = AMDGPU.ROCArray(A)
        AMDGPU.synchronize()
        t = @elapsed begin
            AMDGPU.rocSOLVER.potrf!('L', dA)
            AMDGPU.synchronize()
        end
        return t, Array(dA)
    end
end

function make_runner(o)
    o.impl == "vendor" && return A -> run_vendor(A, o)
    f! = if o.impl == "builtin"
        DA -> LinearAlgebra._chol!(DA, LowerTriangular)
    else
        path = !isempty(o.file)     ? abspath(o.file) :
               o.impl == "student" ? joinpath(ROOT, "exercise", "my_cholesky.jl") :
                                     joinpath(ROOT, "solution", "my_cholesky.jl")
        mod = load_impl(path)
        Base.invokelatest(isdefined, mod, :my_cholesky!) ||
            error("$path must define my_cholesky!(T)")
        impl! = Base.invokelatest(getglobal, mod, :my_cholesky!)
        DA -> Base.invokelatest(impl!, DA.chunks)
    end
    return A -> run_darray(f!, A, o)
end

# ---------------------------------------------------------------------------
# Results

const CSV_HEADER = "timestamp,label,impl,target,nranks,nthreads,n,bs,trial,seconds,gflops,resid,pass,host"

function record!(o, n, trial, t, resid)
    myrank() == 0 || return
    mkpath(dirname(o.out))
    fresh = !isfile(o.out)
    open(o.out, "a") do io
        fresh && println(io, CSV_HEADER)
        println(io, join((Dates.format(now(), "yyyy-mm-ddTHH:MM:SS"), o.label, o.impl,
                          o.target, nranks(), Threads.nthreads(), n, o.bs, trial,
                          @sprintf("%.6f", t), @sprintf("%.2f", cholesky_gflops(n, t)),
                          @sprintf("%.3g", resid), resid < RESID_TOL, gethostname()), ','))
    end
end
