# Demo 5: one matrix product, one Julia process, several GPUs on one node.
# C = A * B is split into block rows; each row runs on a different GPU.
#
#   bash frontier/launch.sh 1 demos/05_multigpu.jl
#   bash frontier/launch.sh 1 demos/05_multigpu.jl --gpus=4

isdefined(@__MODULE__, :DemoPresentation) || include(joinpath(@__DIR__, "presentation.jl"))

# source-begin
using Dagger, Random, LinearAlgebra, AMDGPU

AMDGPU.functional() || error("No functional AMD GPU")

# Number of GPUs to use: `--gpus=N` or `--gpus=all` (default 2).
function gpu_count()
    options = filter(arg -> startswith(arg, "--gpus="), ARGS)
    length(options) <= 1 || error("Specify --gpus only once")
    value = isempty(options) ? "2" : split(only(options), '='; limit=2)[2]
    count = value == "all" ? length(AMDGPU.devices()) : parse(Int, value)
    1 <= count <= length(AMDGPU.devices()) ||
        error("Requested $count GPUs, visible $(length(AMDGPU.devices()))")
    return count
end

# Physical address of a GPU, used to prove that distinct devices were used.
function pci_address(device)
    p = AMDGPU.HIP.properties(device)
    return join((string(p.pciDomainID; base=16, pad=4),
                 string(p.pciBusID; base=16, pad=2),
                 string(p.pciDeviceID; base=16, pad=2)), ':')
end

# Where a task ran: host, process, GPU index and PCI address.
function task_location()
    device = AMDGPU.device()
    return (; host=gethostname(), pid=getpid(),
            gpu=AMDGPU.device_id(device), pci=pci_address(device))
end

# One output tile: C[i] <- A[i] * B, then report where it ran.   (mul!)
function multiply_and_report!(C, A, B)
    mul!(C, A, B)
    AMDGPU.synchronize()
    return task_location()
end

# Spawn one `multiply_and_report!` per block row, pinned to its own GPU.
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
    # (1) One scope per GPU, and one scope that spans all of them.
    ngpus = gpu_count()
    scopes = [Dagger.scope(rocm_gpu=g) for g in 1:ngpus]
    target = Dagger.scope(rocm_gpus=1:ngpus)

    # (2) Run the product tiled across the GPUs.
    reports, result, A, B = matrix_product(scopes, target)

    # (3) Verify the result, and that every GPU was a different device.
    @assert result ≈ A * B
    identities = Set((r.host, r.pci) for r in reports)
    @assert length(identities) == ngpus "GPU tasks reused a physical device"

    # (4) Report where each output tile ran.
    println("One matrix product: A$(size(A)) * B$(size(B))")
    for (i, report) in enumerate(reports)
        println("  output tile row $i -> GPU $(report.gpu) on $(report.host)")
    end
    println("PASS: C ≈ A * B")
    println("GPUs: $ngpus | nodes: $(length(Set(r.host for r in reports))) | Julia processes: 1")
end
# source-end

function run_demo()
    DemoPresentation.present(run_product; title="05 Multiple GPUs",
                             files=[@__FILE__], demo="05_multigpu")
end
if abspath(PROGRAM_FILE) == @__FILE__
    run_demo()
end
