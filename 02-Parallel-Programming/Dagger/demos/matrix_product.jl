# Shared tiled C = A * B demo. MPI placement may use one or several nodes.
using Dagger, Random, LinearAlgebra
if DEMO_USE_GPU
    using AMDGPU
    AMDGPU.functional() || error("No functional AMD GPU")
end
if DEMO_USE_MPI
    using MPI
    Dagger.accelerate!(:mpi)
end
const rank = DEMO_USE_MPI ? MPI.Comm_rank(MPI.COMM_WORLD) : 0
const ranks = DEMO_USE_MPI ? MPI.Comm_size(MPI.COMM_WORLD) : 1
BLAS.set_num_threads(1)

# Record where the actual matrix-multiplication task executes. For GPU work,
# synchronize before reporting and use PCI addresses to distinguish devices
# even when different MPI ranks have the same local numbering.
function multiply_and_report!(C, A, B)
    mul!(C, A, B)
    if DEMO_USE_GPU
        AMDGPU.synchronize()
        device = AMDGPU.device()
        properties = AMDGPU.HIP.properties(device)
        pci = join((string(properties.pciDomainID; base=16, pad=4),
                    string(properties.pciBusID; base=16, pad=2),
                    string(properties.pciDeviceID; base=16, pad=2)), ':')
        identity = (; gpu=AMDGPU.device_id(device), pci)
    else
        identity = (; gpu=0, pci="")
    end
    return (; host=gethostname(), pid=getpid(), rank, identity...)
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

function demonstrate()
    ranks >= DEMO_MIN_RANKS || error("This demo requires at least $DEMO_MIN_RANKS MPI ranks")
    options = filter(a -> startswith(a, "--gpus="), ARGS)
    length(options) <= 1 || error("Specify --gpus only once")
    value = isempty(options) ? DEMO_DEFAULT_GPUS : split(only(options), '='; limit=2)[2]
    ngpus = if DEMO_USE_GPU
        count = value == "all" ? length(AMDGPU.devices()) : parse(Int, value)
        DEMO_MIN_GPUS <= count <= length(AMDGPU.devices()) ||
            error("Need at least $DEMO_MIN_GPUS GPUs per rank; requested $count, visible $(length(AMDGPU.devices()))")
        count
    else
        isempty(options) || error("The CPU demo does not accept --gpus")
        0
    end
    scopes = if DEMO_USE_GPU
        [DEMO_USE_MPI ? Dagger.scope(mpi_rank=r, rocm_gpu=g) : Dagger.scope(rocm_gpu=g)
         for r in 0:ranks-1 for g in 1:ngpus]
    else
        [Dagger.constrain(Dagger.scope(mpi_rank=r), Dagger.DefaultScope()) for r in 0:ranks-1]
    end
    bs = 512
    A = rand(Xoshiro(1), bs * length(scopes), bs)
    B = rand(Xoshiro(2), bs, bs)
    target = DEMO_USE_GPU ? Dagger.scope(rocm_gpus=1:ngpus) :
             Dagger.constrain(Dagger.scope(mpi_ranks=:), Dagger.DefaultScope())
    Dagger.with_options(scope=target) do
        DA = DArray(A, Blocks(bs, bs))
        DB = DArray(B, Blocks(bs, bs))
        DC = DArray(zeros(size(A, 1), size(B, 2)), Blocks(bs, bs))
        reports = tiled_matmul!(DC, DA, DB, scopes)
        result = collect(DC)
        @assert result ≈ A * B
        nodes = length(Set(r.host for r in reports))
        processes = length(Set((r.host, r.pid) for r in reports))
        @assert processes == ranks "Some MPI ranks did not participate in the matrix product"
        identities = DEMO_USE_GPU ? Set((r.host, r.pci) for r in reports) : Set()
        if DEMO_USE_GPU
            @assert length(identities) == length(scopes) "GPU tasks reused a physical device"
        end
        if rank == 0
            println("One matrix product: A$(size(A)) * B$(size(B))")
            for (i, report) in enumerate(reports)
                location = if DEMO_USE_GPU
                    DEMO_USE_MPI ? "rank $(report.rank), local GPU $(report.gpu)" : "GPU $(report.gpu)"
                else
                    "rank $(report.rank), CPU threads"
                end
                println("  output tile row $i -> $location on $(report.host)")
            end
            println("PASS: C ≈ A * B")
            println("GPUs: $(length(identities)) | nodes: $nodes | Julia processes: $processes")
        end
    end
end

demonstrate()
