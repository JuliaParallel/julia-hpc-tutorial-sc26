# Demo 6: the GPU demo's code on MPI ranks (CPU), on one or several nodes.
# Only the scope changes: Dagger spreads the tiles' tasks over the ranks.
#
# The console runs 1 to 32 ranks, one per Odo node (its Ranks menu). In an
# existing allocation, two ranks on its node:
#   bash frontier/multinode.sh cpu

isdefined(@__MODULE__, :DemoPresentation) || include(joinpath(@__DIR__, "presentation.jl"))

# source-begin
using Dagger, MPI, LinearAlgebra, Random

# Every rank runs this same program; `rank` says which one we are.
Dagger.accelerate!(:mpi)
const rank = MPI.Comm_rank(MPI.COMM_WORLD)
const ranks = MPI.Comm_size(MPI.COMM_WORLD)
BLAS.set_num_threads(1)   # one BLAS thread per process

# The same tile operation as the CPU and GPU demos.
double!(X) = (X .*= 2; nothing)

function run_ranks()
    # (1) Choose the scope: the CPUs of every MPI rank.
    cpus = Dagger.constrain(Dagger.scope(mpi_ranks=:), Dagger.DefaultScope())

    Dagger.with_options(scope=cpus) do
        # (2) Tile the array and double every tile; the same code again. The
        #     input is seeded, so every rank builds the same array.
        DA = DArray(rand(Xoshiro(1), 4096, 4096), Blocks(1024, 1024))
        A0 = collect(DA)
        Dagger.spawn_datadeps() do
            for tile in DA.chunks
                Dagger.@spawn double!(InOut(tile))
            end
        end
        doubled = collect(DA) ≈ 2 .* A0
        rank == 0 && println("doubled on $ranks ranks: ", doubled)
        @assert doubled

        # (3) DArray matmul, now across the ranks.
        DC = DA * DA
        host_A = collect(DA)
        product_correct = collect(DC) ≈ host_A * host_A
        rank == 0 && println("matmul on $ranks ranks correct: ", product_correct)
        @assert product_correct
    end
end
# source-end

# ---- Not shown to attendees: process check. ----

# The Julia process a task runs in.
process_identity() = (gethostname(), getpid())

# Every rank in the scope is its own Julia process.
function check_processes()
    scopes = [Dagger.constrain(Dagger.scope(mpi_rank=r), Dagger.DefaultScope()) for r in 0:ranks-1]
    processes = Set(fetch(Dagger.@spawn scope=s process_identity()) for s in scopes)
    @assert length(processes) == ranks "Some MPI ranks did not participate"
    rank == 0 && println("distinct processes in the scope: ", length(processes))
end

function run_demo()
    DemoPresentation.present(; title="06 MPI ranks", files=[@__FILE__],
                             demo="06_multinode", mpi=MPI) do
        run_ranks()
        check_processes()
    end
end
if abspath(PROGRAM_FILE) == @__FILE__
    run_demo()
end
