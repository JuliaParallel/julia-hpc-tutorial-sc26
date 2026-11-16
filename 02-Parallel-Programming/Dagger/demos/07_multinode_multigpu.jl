# Demo 7: the GPU demo's code on MPI ranks with GPUs, on one or several nodes.
# Only the scope changes: Dagger spreads the tiles' tasks over every GPU of
# every rank.
#
# The console runs 1 to 32 ranks, one per Odo node with all its eight GPUs
# (its Ranks menu). In an existing allocation, two ranks with two GPUs each:
#   bash frontier/multinode.sh gpu

isdefined(@__MODULE__, :DemoPresentation) || include(joinpath(@__DIR__, "presentation.jl"))

# source-begin
using Dagger, MPI, AMDGPU, LinearAlgebra, Random

# Every rank runs this same program; `rank` says which one we are.
Dagger.accelerate!(:mpi)
const rank = MPI.Comm_rank(MPI.COMM_WORLD)
const ranks = MPI.Comm_size(MPI.COMM_WORLD)

# The same tile operation as the CPU and GPU demos.
double!(X) = (X .*= 2; nothing)

function run_ranks_and_gpus()
    # (1) Choose the scope: every GPU of every MPI rank.
    gpus = Dagger.scope(rocm_gpus=:)

    Dagger.with_options(scope=gpus) do
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
        rank == 0 && println("doubled on $ranks ranks' GPUs: ", doubled)
        @assert doubled

        # (3) DArray matmul, now across every rank's GPUs.
        DC = DA * DA
        host_A = collect(DA)
        product_correct = collect(DC) ≈ host_A * host_A
        rank == 0 && println("matmul on $ranks ranks' GPUs correct: ", product_correct)
        @assert product_correct
    end
end
# source-end

# ---- Not shown to attendees: GPU check and device check. ----

AMDGPU.functional() || error("No functional AMD GPU")
const ngpus = length(AMDGPU.devices())   # GPUs per rank

# The physical device and process a task runs on.
function device_identity()
    p = AMDGPU.HIP.properties(AMDGPU.device())
    return (gethostname(), getpid(),
            join((string(p.pciDomainID; base=16, pad=4),
                  string(p.pciBusID; base=16, pad=2),
                  string(p.pciDeviceID; base=16, pad=2)), ':'))
end

# Every (rank, GPU) pair in the scope is its own physical GPU, in every rank.
function check_devices()
    found = [fetch(Dagger.@spawn scope=Dagger.scope(mpi_rank=r, rocm_gpu=g) device_identity())
             for r in 0:ranks-1 for g in 1:ngpus]
    @assert length(Set((h, pci) for (h, _, pci) in found)) == length(found) "GPU scopes reused a physical device"
    @assert length(Set((h, pid) for (h, pid, _) in found)) == ranks "Some MPI ranks did not participate"
    rank == 0 && println("distinct GPUs in the scope: ", length(found), " ($ranks ranks x $ngpus)")
end

function run_demo()
    DemoPresentation.present(; title="07 MPI ranks and GPUs", files=[@__FILE__],
                             demo="07_multinode_multigpu", mpi=MPI) do
        run_ranks_and_gpus()
        check_devices()
    end
end
if abspath(PROGRAM_FILE) == @__FILE__
    run_demo()
end
