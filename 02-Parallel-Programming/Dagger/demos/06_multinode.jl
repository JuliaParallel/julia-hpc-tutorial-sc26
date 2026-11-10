# Demo 6: one CPU matrix product across MPI ranks, on one or several nodes.
# C = A * B is split into block rows; each row runs on a different rank.
#
# Odo tutorial: two ranks on the existing node.
#   bash frontier/multinode.sh cpu

isdefined(@__MODULE__, :DemoPresentation) || include(joinpath(@__DIR__, "presentation.jl"))

# source-begin
using Dagger, MPI, Random, LinearAlgebra

# Every rank runs this same program; `rank` says which one we are.
Dagger.accelerate!(:mpi)
const rank = MPI.Comm_rank(MPI.COMM_WORLD)
const ranks = MPI.Comm_size(MPI.COMM_WORLD)
ranks >= 2 || error("This demo needs at least two MPI ranks")
BLAS.set_num_threads(1)   # one BLAS thread per process

# Where a task ran: host, process and MPI rank.
function task_location()
    return (; host=gethostname(), pid=getpid(), rank)
end

# One output tile: C[i] <- A[i] * B, then report where it ran.   (mul!)
function multiply_and_report!(C, A, B)
    mul!(C, A, B)
    return task_location()
end

# Spawn one `multiply_and_report!` per block row, pinned to its own rank.
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

# Build A (one block row per scope), B and C as tiled DArrays; run the product.
function matrix_product(scopes, target)
    bs = 512
    A = rand(Xoshiro(1), bs * length(scopes), bs)
    B = rand(Xoshiro(2), bs, bs)
    Dagger.with_options(scope=target) do
        DA = DArray(A, Blocks(bs, bs))
        DB = DArray(B, Blocks(bs, bs))
        DC = DArray(zeros(size(A, 1), size(B, 2)), Blocks(bs, bs))
        reports = tiled_matmul!(DC, DA, DB, scopes)
        return reports, collect(DC), A, B
    end
end

function run_product()
    # (1) One scope per rank, and one scope that spans all of them.
    scopes = [Dagger.constrain(Dagger.scope(mpi_rank=r), Dagger.DefaultScope())
              for r in 0:ranks-1]
    target = Dagger.constrain(Dagger.scope(mpi_ranks=:), Dagger.DefaultScope())

    # (2) Run the product tiled across the ranks.
    reports, result, A, B = matrix_product(scopes, target)

    # (3) Verify the result, and that every rank took part.
    @assert result ≈ A * B
    processes = length(Set((r.host, r.pid) for r in reports))
    @assert processes == ranks "Some MPI ranks did not participate in the matrix product"

    # (4) Report where each output tile ran (rank 0 prints).
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
