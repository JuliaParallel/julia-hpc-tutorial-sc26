# Demo 5: the single-GPU demo on several GPUs of one node. The code is demo 3's;
# only the scope changes, so Dagger spreads the tiles' tasks over the GPUs.
#
#   bash frontier/launch.sh 1 demos/05_multigpu.jl
#   bash frontier/launch.sh 1 demos/05_multigpu.jl --gpus=4   (or 8: a whole Odo node)

isdefined(@__MODULE__, :DemoPresentation) || include(joinpath(@__DIR__, "presentation.jl"))

# source-begin
using Dagger, AMDGPU, LinearAlgebra, Random

# The same tile operation as the CPU and GPU demos.
double!(X) = (X .*= 2; nothing)

function run_gpus()
    # (1) Choose the scope: several GPUs instead of one. Every task spawned
    #     inside runs on one of them.
    gpus = Dagger.scope(rocm_gpus=1:gpu_count())

    Dagger.with_options(scope=gpus) do
        # (2) Tile the array and double every tile; this is demo 3, unchanged.
        #     The 16 tiles' tasks are spread over the GPUs.
        DA = DArray(rand(Xoshiro(1), 4096, 4096), Blocks(1024, 1024))
        A0 = collect(DA)
        Dagger.spawn_datadeps() do
            for tile in DA.chunks
                Dagger.@spawn double!(InOut(tile))
            end
        end
        doubled = collect(DA) ≈ 2 .* A0
        println("doubled on ", gpu_count(), " GPUs: ", doubled)
        @assert doubled

        # (3) DArray matmul, now over every GPU in the scope.
        DC = DA * DA
        host_A = collect(DA)
        product_correct = collect(DC) ≈ host_A * host_A
        println("matmul on ", gpu_count(), " GPUs correct: ", product_correct)
        @assert product_correct
    end
end
# source-end

# ---- Not shown to attendees: argument parsing and device check. ----

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

# The physical device a task runs on, by host and PCI address.
function device_identity()
    p = AMDGPU.HIP.properties(AMDGPU.device())
    return (gethostname(), join((string(p.pciDomainID; base=16, pad=4),
                                 string(p.pciBusID; base=16, pad=2),
                                 string(p.pciDeviceID; base=16, pad=2)), ':'))
end

# Every GPU in the scope is a distinct physical device.
function check_devices()
    devices = Set(fetch(Dagger.@spawn scope=Dagger.scope(rocm_gpu=g) device_identity())
                  for g in 1:gpu_count())
    @assert length(devices) == gpu_count() "GPU scopes reused a physical device"
    println("distinct GPUs in the scope: ", length(devices))
end

function run_demo()
    DemoPresentation.present(; title="05 Multiple GPUs", files=[@__FILE__],
                             demo="05_multigpu") do
        run_gpus()
        check_devices()
    end
end
if abspath(PROGRAM_FILE) == @__FILE__
    run_demo()
end
