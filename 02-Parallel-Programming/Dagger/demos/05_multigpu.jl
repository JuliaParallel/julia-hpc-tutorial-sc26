# Demo 5: one matrix product, one Julia process, several GPUs on one node.
# C = A * B is split into block rows; each row runs on a different GPU.
#
#   bash andromeda/launch.sh 1 2 demos/05_multigpu.jl
#   bash andromeda/launch.sh 1 1 demos/05_multigpu.jl --gpus=1

isdefined(@__MODULE__, :DemoPresentation) || include(joinpath(@__DIR__, "presentation.jl"))

# source-begin
using Dagger, LinearAlgebra, Random, CUDA

function run_product()
    ngpus = gpu_count()
    bs = 512

    # (1) One scope per GPU, and one scope that spans all of them.
    scopes = [Dagger.scope(cuda_gpu=g) for g in 1:ngpus]
    all_gpus = Dagger.scope(cuda_gpus=1:ngpus)

    # (2) A holds one tile per GPU; B is a single tile.
    A = rand(Xoshiro(1), bs * ngpus, bs)
    B = rand(Xoshiro(2), bs, bs)

    C = Dagger.with_options(scope=all_gpus) do
        DA = DArray(A, Blocks(bs, bs))
        DB = DArray(B, Blocks(bs, bs))
        DC = DArray(zeros(bs * ngpus, bs), Blocks(bs, bs))

        # (3) C[i] <- A[i] * B, with tile i pinned to GPU i.
        Dagger.spawn_datadeps() do
            for (i, gpu) in enumerate(scopes)
                Dagger.@spawn scope=gpu mul!(Out(DC.chunks[i, 1]), In(DA.chunks[i, 1]), In(DB.chunks[1, 1]))
            end
        end
        collect(DC)
    end

    @assert C ≈ A * B
    println("PASS: C ≈ A * B on $ngpus GPUs")
    return scopes
end
# source-end

# ---- Not shown to attendees: argument parsing and placement check. ----

CUDA.functional() || error("No functional NVIDIA GPU")

# Number of GPUs to use: `--gpus=N` or `--gpus=all` (default 2).
function gpu_count()
    options = filter(arg -> startswith(arg, "--gpus="), ARGS)
    length(options) <= 1 || error("Specify --gpus only once")
    value = isempty(options) ? "2" : split(only(options), '='; limit=2)[2]
    count = value == "all" ? length(CUDA.devices()) : parse(Int, value)
    1 <= count <= length(CUDA.devices()) ||
        error("Requested $count GPUs, visible $(length(CUDA.devices()))")
    return count
end

# The physical device a task runs on, by host and device UUID.
device_identity() = (gethostname(), string(CUDA.uuid(CUDA.device())))

# Every scope must land on its own physical GPU.
function check_placement(scopes)
    devices = Set(fetch(Dagger.@spawn scope=s device_identity()) for s in scopes)
    @assert length(devices) == length(scopes) "GPU scopes reused a physical device"
    println("distinct GPUs used: ", length(devices))
end

function run_demo()
    DemoPresentation.present(; title="05 Multiple GPUs", files=[@__FILE__],
                             demo="05_multigpu") do
        check_placement(run_product())
    end
end
if abspath(PROGRAM_FILE) == @__FILE__
    run_demo()
end
