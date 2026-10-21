# One matrix multiplication split across distinct GPUs.
# Default: one Julia process and two GPUs on the same node.
#   bash frontier/launch.sh 1 demos/05_multigpu.jl
# Optional same-node MPI check, with one GPU per rank:
#   bash frontier/launch.sh 4 demos/05_multigpu.jl --mpi
using Dagger, AMDGPU, Random, LinearAlgebra
const USE_MPI = "--mpi" in ARGS
if USE_MPI
    using MPI
    Dagger.accelerate!(:mpi)
end
const rank = USE_MPI ? MPI.Comm_rank(MPI.COMM_WORLD) : 0
const ranks = USE_MPI ? MPI.Comm_size(MPI.COMM_WORLD) : 1
AMDGPU.functional() || error("No functional AMD GPU")
BLAS.set_num_threads(1)

# Record the device from inside the actual matrix-multiplication task, after
# its GPU work completes. The PCI address distinguishes physical GPU devices
# even when different MPI ranks have the same local device numbering.
function multiply_and_report!(C, A, B)
    mul!(C, A, B)
    AMDGPU.synchronize()
    device = AMDGPU.device()
    properties = AMDGPU.HIP.properties(device)
    pci = join((string(properties.pciDomainID; base=16, pad=4),
                string(properties.pciBusID; base=16, pad=2),
                string(properties.pciDeviceID; base=16, pad=2)), ':')
    return (; host=gethostname(), pid=getpid(), rank, gpu=AMDGPU.device_id(device), pci)
end

function tiled_matmul!(DC, DA, DB, scopes)
    tasks = []
    Dagger.spawn_datadeps() do
        for (i, gpu) in enumerate(scopes)
            task = Dagger.@spawn scope=gpu multiply_and_report!(
                Out(DC.chunks[i, 1]), In(DA.chunks[i, 1]), In(DB.chunks[1, 1]))
            push!(tasks, task)
        end
    end
    return fetch.(tasks)
end

function demonstrate()
    options = filter(a -> startswith(a, "--gpus="), ARGS)
    length(options) <= 1 || error("Specify --gpus only once")
    value = isempty(options) ? (USE_MPI ? "all" : "2") : split(only(options), '='; limit=2)[2]
    ngpus = value == "all" ? length(AMDGPU.devices()) : parse(Int, value)
    1 <= ngpus <= length(AMDGPU.devices()) ||
        error("Requested $ngpus GPUs per process, but $(length(AMDGPU.devices())) are visible")
    scopes = [USE_MPI ? Dagger.scope(mpi_rank=r, rocm_gpu=g) : Dagger.scope(rocm_gpu=g)
              for r in 0:ranks-1 for g in 1:ngpus]
    bs = 512
    A = rand(Xoshiro(1), bs * length(scopes), bs)
    B = rand(Xoshiro(2), bs, bs)
    gpu = Dagger.scope(rocm_gpus=1:ngpus)
    Dagger.with_options(scope=gpu) do
        DA = DArray(A, Blocks(bs, bs))
        DB = DArray(B, Blocks(bs, bs))
        DC = DArray(zeros(size(A, 1), size(B, 2)), Blocks(bs, bs))
        reports = tiled_matmul!(DC, DA, DB, scopes)
        result = collect(DC)
        @assert result ≈ A * B
        identities = Set((r.host, r.pci) for r in reports)
        @assert length(identities) == length(scopes) "GPU tasks reused a physical device"
        if rank == 0
            println("One matrix product: A$(size(A)) * B$(size(B))")
            for (i, report) in enumerate(reports)
                location = USE_MPI ? "rank $(report.rank), local GPU $(report.gpu)" : "GPU $(report.gpu)"
                println("  output tile row $i -> $location on $(report.host)")
            end
            nodes = length(Set(r.host for r in reports))
            processes = length(Set((r.host, r.pid) for r in reports))
            println("PASS: C ≈ A * B")
            println("GPUs: $(length(identities)) | nodes: $nodes | Julia processes: $processes")
        end
    end
end

demonstrate()
