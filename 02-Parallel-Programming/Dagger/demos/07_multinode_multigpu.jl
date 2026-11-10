# One MPI matrix product with two distinct GPUs per rank, on one or several nodes.
# Odo tutorial: two ranks on the existing node. bash frontier/multinode.sh gpu

isdefined(@__MODULE__, :DemoPresentation) || include(joinpath(@__DIR__, "presentation.jl"))

# source-begin
# Setup: MPI, rank identity, and two GPUs per rank.
using Dagger, MPI, AMDGPU, Random, LinearAlgebra

Dagger.accelerate!(:mpi)
const rank = MPI.Comm_rank(MPI.COMM_WORLD)
const ranks = MPI.Comm_size(MPI.COMM_WORLD)
ranks >= 2 || error("This demo needs at least two MPI ranks")
AMDGPU.functional() || error("No functional AMD GPU")
const ngpus = 2
length(AMDGPU.devices()) >= ngpus ||
    error("This demo needs $ngpus GPUs per rank, visible $(length(AMDGPU.devices()))")
# source-end

# source-begin
# Algorithm: each rank computes one tile row per local GPU.
function task_location()
    device = AMDGPU.device()
    properties = AMDGPU.HIP.properties(device)
    pci = join((string(properties.pciDomainID; base=16, pad=4),
                string(properties.pciBusID; base=16, pad=2),
                string(properties.pciDeviceID; base=16, pad=2)), ':')
    return (; host=gethostname(), pid=getpid(), rank,
            gpu=AMDGPU.device_id(device), pci)
end

function multiply_and_report!(C, A, B)
    mul!(C, A, B)
    AMDGPU.synchronize()
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
# Setup and use: run across ranks and GPUs, then verify every device participated.
function run_product()
    scopes = [Dagger.scope(mpi_rank=r, rocm_gpu=g)
              for r in 0:ranks-1 for g in 1:ngpus]
    target = Dagger.scope(rocm_gpus=1:ngpus)
    reports, result, A, B = matrix_product(scopes, target)

    @assert result ≈ A * B
    processes = length(Set((r.host, r.pid) for r in reports))
    @assert processes == ranks "Some MPI ranks did not participate in the matrix product"
    identities = Set((r.host, r.pci) for r in reports)
    @assert length(identities) == length(scopes) "GPU tasks reused a physical device"

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
