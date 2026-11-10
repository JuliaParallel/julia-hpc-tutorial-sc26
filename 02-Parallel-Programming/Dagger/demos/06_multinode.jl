# One CPU matrix product across MPI ranks, on one or several nodes.
# Odo tutorial: two ranks on the existing node. bash frontier/multinode.sh cpu

isdefined(@__MODULE__, :DemoPresentation) || include(joinpath(@__DIR__, "presentation.jl"))

# source-begin
# Setup: MPI, rank identity, and one BLAS thread per process.
using Dagger, MPI, Random, LinearAlgebra

Dagger.accelerate!(:mpi)
const rank = MPI.Comm_rank(MPI.COMM_WORLD)
const ranks = MPI.Comm_size(MPI.COMM_WORLD)
ranks >= 2 || error("This demo needs at least two MPI ranks")
BLAS.set_num_threads(1)
# source-end

# source-begin
# Algorithm: each rank computes one output tile row.
function task_location()
    return (; host=gethostname(), pid=getpid(), rank)
end

function multiply_and_report!(C, A, B)
    mul!(C, A, B)
    return task_location()
end

function tiled_matmul!(DC, DA, DB, scopes)
    tasks = []
    Dagger.spawn_datadeps() do
        for (i, placement) in enumerate(scopes)
            task = Dagger.@spawn scope=placement multiply_and_report!(
                Out(DC.chunks[i, 1]), In(DA.chunks[i, 1]), In(DB.chunks[1, 1]))
            push!(tasks, task)
        end
    end
    return fetch.(tasks)
end

function matrix_product(scopes, target)
    bs = 512
    A = rand(Xoshiro(1), bs * length(scopes), bs)
    B = rand(Xoshiro(2), bs, bs)
    Dagger.with_options(scope=target) do
        DA = DArray(A, Blocks(bs, bs))
        DB = DArray(B, Blocks(bs, bs))
        DC = DArray(zeros(size(A, 1), size(B, 2)), Blocks(bs, bs))
        reports = tiled_matmul!(DC, DA, DB, scopes)
        result = collect(DC)
        return reports, result, A, B
    end
end
# source-end

# source-begin
# Setup and use: run across ranks and report where the output tiles ran.
function run_product()
    scopes = [Dagger.constrain(Dagger.scope(mpi_rank=r), Dagger.DefaultScope())
              for r in 0:ranks-1]
    target = Dagger.constrain(Dagger.scope(mpi_ranks=:), Dagger.DefaultScope())
    reports, result, A, B = matrix_product(scopes, target)

    @assert result ≈ A * B
    processes = length(Set((r.host, r.pid) for r in reports))
    @assert processes == ranks "Some MPI ranks did not participate in the matrix product"

    if rank == 0
        println("One matrix product: A$(size(A)) * B$(size(B))")
        for (i, report) in enumerate(reports)
            println("  output tile row $i -> rank $(report.rank) on $(report.host)")
        end
        println("PASS: C ≈ A * B")
        println("nodes: $(length(Set(r.host for r in reports))) | Julia processes: $processes")
    end
end
# source-end

function run_demo()
    DemoPresentation.present(run_product; title="06 MPI matrix product",
                             files=[@__FILE__], demo="06_multinode", mpi=MPI)
end
if abspath(PROGRAM_FILE) == @__FILE__
    run_demo()
end
