# Demo 7: one MPI matrix product with two distinct GPUs per rank, on one or
# several nodes. C = A * B is split into block rows; each row runs on its own
# (rank, GPU) pair.
#
# Odo tutorial: two ranks on the existing node.
#   bash frontier/multinode.sh gpu

isdefined(@__MODULE__, :DemoPresentation) || include(joinpath(@__DIR__, "presentation.jl"))

# source-begin
using Dagger, MPI, AMDGPU, Random, LinearAlgebra

# Every rank runs this same program; `rank` says which one we are.
Dagger.accelerate!(:mpi)
const rank = MPI.Comm_rank(MPI.COMM_WORLD)
const ranks = MPI.Comm_size(MPI.COMM_WORLD)
ranks >= 2 || error("This demo needs at least two MPI ranks")
AMDGPU.functional() || error("No functional AMD GPU")
const ngpus = 2   # GPUs per rank
length(AMDGPU.devices()) >= ngpus ||
    error("This demo needs $ngpus GPUs per rank, visible $(length(AMDGPU.devices()))")

# Physical address of a GPU, used to prove that distinct devices were used.
function pci_address(device)
    p = AMDGPU.HIP.properties(device)
    return join((string(p.pciDomainID; base=16, pad=4),
                 string(p.pciBusID; base=16, pad=2),
                 string(p.pciDeviceID; base=16, pad=2)), ':')
end

# Where a task ran: host, process, MPI rank, GPU index and PCI address.
function task_location()
    device = AMDGPU.device()
    return (; host=gethostname(), pid=getpid(), rank,
            gpu=AMDGPU.device_id(device), pci=pci_address(device))
end

# One output tile: C[i] <- A[i] * B, then report where it ran.   (mul!)
function multiply_and_report!(C, A, B)
    mul!(C, A, B)
    AMDGPU.synchronize()
    return task_location()
end

# Spawn one `multiply_and_report!` per block row, pinned to its own rank and GPU.
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
    # (1) One scope per (rank, GPU) pair, and one scope that spans all GPUs.
    scopes = [Dagger.scope(mpi_rank=r, rocm_gpu=g)
              for r in 0:ranks-1 for g in 1:ngpus]
    target = Dagger.scope(rocm_gpus=1:ngpus)

    # (2) Run the product tiled across the ranks and GPUs.
    reports, result, A, B = matrix_product(scopes, target)

    # (3) Verify the result, that every rank took part, and that every GPU
    #     was a different device.
    @assert result ≈ A * B
    processes = length(Set((r.host, r.pid) for r in reports))
    @assert processes == ranks "Some MPI ranks did not participate in the matrix product"
    identities = Set((r.host, r.pci) for r in reports)
    @assert length(identities) == length(scopes) "GPU tasks reused a physical device"

    # (4) Report where each output tile ran (rank 0 prints).
    if rank == 0
        println("One matrix product: A$(size(A)) * B$(size(B))")
        for (i, report) in enumerate(reports)
            println("  output tile row $i -> rank $(report.rank), " *
                    "GPU $(report.gpu) on $(report.host)")
        end
        println("PASS: C ≈ A * B")
        println("GPUs: $(length(identities)) | " *
                "nodes: $(length(Set(r.host for r in reports))) | " *
                "Julia processes: $processes")
    end
end
# source-end

function run_demo()
    DemoPresentation.present(run_product; title="07 MPI and multiple GPUs",
                             files=[@__FILE__], demo="07_multinode_multigpu", mpi=MPI)
end
if abspath(PROGRAM_FILE) == @__FILE__
    run_demo()
end
