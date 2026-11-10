# Demo 6: one CPU matrix product across MPI ranks, on one or several nodes.
# C = A * B is split into block rows; each row runs on a different rank.
#
# Odo tutorial: two ranks on the existing node.
#   bash frontier/multinode.sh cpu

isdefined(@__MODULE__, :DemoPresentation) || include(joinpath(@__DIR__, "presentation.jl"))

# source-begin
using Dagger, MPI, LinearAlgebra, Random

# Every rank runs this same program; `rank` says which one we are.
Dagger.accelerate!(:mpi)
const rank = MPI.Comm_rank(MPI.COMM_WORLD)
const ranks = MPI.Comm_size(MPI.COMM_WORLD)
BLAS.set_num_threads(1)   # one BLAS thread per process

function run_product()
    bs = 512

    # (1) One scope per rank, and one scope that spans all of them.
    scopes = [Dagger.constrain(Dagger.scope(mpi_rank=r), Dagger.DefaultScope())
              for r in 0:ranks-1]
    all_ranks = Dagger.constrain(Dagger.scope(mpi_ranks=:), Dagger.DefaultScope())

    # (2) A holds one tile per rank; B is a single tile.
    A = rand(Xoshiro(1), bs * ranks, bs)
    B = rand(Xoshiro(2), bs, bs)

    C = Dagger.with_options(scope=all_ranks) do
        DA = DArray(A, Blocks(bs, bs))
        DB = DArray(B, Blocks(bs, bs))
        DC = DArray(zeros(bs * ranks, bs), Blocks(bs, bs))

        # (3) C[i] <- A[i] * B, with tile i pinned to rank i-1.
        Dagger.spawn_datadeps() do
            for (i, owner) in enumerate(scopes)
                Dagger.@spawn scope=owner mul!(Out(DC.chunks[i, 1]), In(DA.chunks[i, 1]), In(DB.chunks[1, 1]))
            end
        end
        collect(DC)
    end

    @assert C ≈ A * B
    rank == 0 && println("PASS: C ≈ A * B on $ranks ranks")
    return scopes
end
# source-end

# ---- Not shown to attendees: input check and placement check. ----

ranks >= 2 || error("This demo needs at least two MPI ranks")

# The Julia process a task runs in.
process_identity() = (gethostname(), getpid())

# Every scope must land on its own Julia process.
function check_placement(scopes)
    processes = Set(fetch(Dagger.@spawn scope=s process_identity()) for s in scopes)
    @assert length(processes) == ranks "Some MPI ranks did not participate"
    rank == 0 && println("distinct processes used: ", length(processes))
end

function run_demo()
    DemoPresentation.present(; title="06 MPI matrix product", files=[@__FILE__],
                             demo="06_multinode", mpi=MPI) do
        check_placement(run_product())
    end
end
if abspath(PROGRAM_FILE) == @__FILE__
    run_demo()
end
