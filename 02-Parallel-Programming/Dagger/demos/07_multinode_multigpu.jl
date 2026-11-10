# Demo 7: one MPI matrix product with two distinct GPUs per rank, on one or
# several nodes. C = A * B is split into block rows; each row runs on its own
# (rank, GPU) pair.
#
# Odo tutorial: two ranks on the existing node.
#   bash frontier/multinode.sh gpu

isdefined(@__MODULE__, :DemoPresentation) || include(joinpath(@__DIR__, "presentation.jl"))

# source-begin
using Dagger, MPI, AMDGPU, LinearAlgebra, Random

# Every rank runs this same program; `rank` says which one we are.
Dagger.accelerate!(:mpi)
const rank = MPI.Comm_rank(MPI.COMM_WORLD)
const ranks = MPI.Comm_size(MPI.COMM_WORLD)
const ngpus = 2   # GPUs per rank

function run_product()
    bs = 512

    # (1) One scope per (rank, GPU) pair, and one scope that spans all GPUs.
    scopes = [Dagger.scope(mpi_rank=r, rocm_gpu=g) for r in 0:ranks-1 for g in 1:ngpus]
    all_gpus = Dagger.scope(rocm_gpus=1:ngpus)

    # (2) A holds one tile per (rank, GPU) pair; B is a single tile.
    A = rand(Xoshiro(1), bs * length(scopes), bs)
    B = rand(Xoshiro(2), bs, bs)

    C = Dagger.with_options(scope=all_gpus) do
        DA = DArray(A, Blocks(bs, bs))
        DB = DArray(B, Blocks(bs, bs))
        DC = DArray(zeros(bs * length(scopes), bs), Blocks(bs, bs))

        # (3) C[i] <- A[i] * B, with tile i pinned to its own rank and GPU.
        Dagger.spawn_datadeps() do
            for (i, gpu) in enumerate(scopes)
                Dagger.@spawn scope=gpu mul!(Out(DC.chunks[i, 1]), In(DA.chunks[i, 1]), In(DB.chunks[1, 1]))
            end
        end
        collect(DC)
    end

    @assert C ≈ A * B
    rank == 0 && println("PASS: C ≈ A * B on $ranks ranks x $ngpus GPUs")
    return scopes
end
# source-end

# ---- Not shown to attendees: input checks and placement check. ----

ranks >= 2 || error("This demo needs at least two MPI ranks")
AMDGPU.functional() || error("No functional AMD GPU")
length(AMDGPU.devices()) >= ngpus ||
    error("This demo needs $ngpus GPUs per rank, visible $(length(AMDGPU.devices()))")

# The physical device and process a task runs on.
function device_identity()
    p = AMDGPU.HIP.properties(AMDGPU.device())
    return (gethostname(), getpid(),
            join((string(p.pciDomainID; base=16, pad=4),
                  string(p.pciBusID; base=16, pad=2),
                  string(p.pciDeviceID; base=16, pad=2)), ':'))
end

# Every scope must land on its own physical GPU, in every rank.
function check_placement(scopes)
    found = [fetch(Dagger.@spawn scope=s device_identity()) for s in scopes]
    @assert length(Set((h, pci) for (h, _, pci) in found)) == length(scopes) "GPU scopes reused a physical device"
    @assert length(Set((h, pid) for (h, pid, _) in found)) == ranks "Some MPI ranks did not participate"
    rank == 0 && println("distinct GPUs used: ", length(scopes))
end

function run_demo()
    DemoPresentation.present(; title="07 MPI and multiple GPUs", files=[@__FILE__],
                             demo="07_multinode_multigpu", mpi=MPI) do
        check_placement(run_product())
    end
end
if abspath(PROGRAM_FILE) == @__FILE__
    run_demo()
end
