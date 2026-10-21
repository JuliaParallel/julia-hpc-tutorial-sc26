# GPU topology and explicit placement. Run with one process for multi-GPU,
# or with --mpi across ranks on one node. Every selected GPU gets a tile task.
# SC26_GPUS_PER_TASK=2 bash frontier/launch.sh 1 demos/05_multigpu.jl
# SC26_NODES=1 bash frontier/launch.sh 4 demos/05_multigpu.jl --mpi
using Dagger, AMDGPU, Random, LinearAlgebra
const USE_MPI = "--mpi" in ARGS
if USE_MPI
    using MPI
    Dagger.accelerate!(:mpi)
end
const rank = USE_MPI ? MPI.Comm_rank(MPI.COMM_WORLD) : 0
const ranks = USE_MPI ? MPI.Comm_size(MPI.COMM_WORLD) : 1
const devices = length(AMDGPU.devices())
AMDGPU.functional() || error("No functional AMD GPU")
BLAS.set_num_threads(1)
println("rank $rank/$ranks on $(gethostname()): $devices visible GPU(s), $(AMDGPU.devices())")
println("rank $rank: ROCR_VISIBLE_DEVICES=$(get(ENV, "ROCR_VISIBLE_DEVICES", "all"))")

function double_and_report!(X)
    X .*= 2
    write(stdout, "  tile on $(gethostname()), rank $rank, device $(AMDGPU.device()), array $(typeof(X))\n")
    nothing
end

function demonstrate()
    # All ranks see the same device count with our one-GPU-per-rank MPI launch.
    scopes = [USE_MPI ? Dagger.scope(mpi_rank=r, rocm_gpu=g) : Dagger.scope(rocm_gpu=g)
              for r in 0:ranks-1 for g in 1:devices]
    gpu = Dagger.scope(rocm_gpus=:)
    A = rand(Xoshiro(1), 256, 256 * length(scopes))
    Dagger.with_options(scope=gpu) do
        DA = DArray(A, Blocks(256, 256))
        Dagger.spawn_datadeps() do
            for (i, tile) in enumerate(DA.chunks)
                Dagger.@spawn scope=scopes[i] double_and_report!(InOut(tile))
            end
        end
        result = Array(collect(DA))
        @assert result ≈ 2 .* A
        rank == 0 && println("PASS: $(length(scopes)) GPU placements across $ranks rank(s)")
    end
end
demonstrate()
