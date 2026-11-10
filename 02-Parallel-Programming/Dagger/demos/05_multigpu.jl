# One matrix product, one Julia process, several GPUs on one node.
#   bash frontier/launch.sh 1 demos/05_multigpu.jl
#   bash frontier/launch.sh 1 demos/05_multigpu.jl --gpus=4

isdefined(@__MODULE__, :DemoPresentation) || include(joinpath(@__DIR__, "presentation.jl"))

# source-begin
# Setup: packages and the number of GPUs.
using Dagger, Random, LinearAlgebra, AMDGPU

AMDGPU.functional() || error("No functional AMD GPU")

function gpu_count()
    options = filter(arg -> startswith(arg, "--gpus="), ARGS)
    length(options) <= 1 || error("Specify --gpus only once")
    value = isempty(options) ? "2" : split(only(options), '='; limit=2)[2]
    count = value == "all" ? length(AMDGPU.devices()) : parse(Int, value)
    1 <= count <= length(AMDGPU.devices()) ||
        error("Requested $count GPUs, visible $(length(AMDGPU.devices()))")
    return count
end

# source-end

# source-begin
# Algorithm: one BLAS call per output tile, each pinned to a different GPU.
function task_location()
    device = AMDGPU.device()
    properties = AMDGPU.HIP.properties(device)
    pci = join((string(properties.pciDomainID; base=16, pad=4),
                string(properties.pciBusID; base=16, pad=2),
                string(properties.pciDeviceID; base=16, pad=2)), ':')
    return (; host=gethostname(), pid=getpid(), gpu=AMDGPU.device_id(device), pci)
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
# Setup and use: tile the product across the GPUs and verify the result.
function run_product()
    ngpus = gpu_count()
    scopes = [Dagger.scope(rocm_gpu=g) for g in 1:ngpus]
    target = Dagger.scope(rocm_gpus=1:ngpus)
    reports, result, A, B = matrix_product(scopes, target)

    @assert result ≈ A * B
    identities = Set((r.host, r.pci) for r in reports)
    @assert length(identities) == ngpus "GPU tasks reused a physical device"

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
