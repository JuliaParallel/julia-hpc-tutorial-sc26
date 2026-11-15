# Demo 3: the SAME Datadeps code on an AMD GPU. Only the scope changes:
# Dagger moves the tiles to the GPU and swaps BLAS for rocBLAS for you.
#
#   julia --project -t 4 demos/03_gpu.jl

isdefined(@__MODULE__, :DemoPresentation) || include(joinpath(@__DIR__, "presentation.jl"))

# source-begin
using Dagger, AMDGPU, LinearAlgebra

# The same tile operation as the CPU demo.
double!(X) = (X .*= 2; nothing)

function run_gpu()
    # (1) Choose the GPU scope: every task spawned inside runs on one GPU.
    gpu = Dagger.scope(rocm_gpu=1)

    Dagger.with_options(scope=gpu) do
        # (2) Tile the array and double every tile; this is demo 2, unchanged.
        DA = DArray(rand(4096, 4096), Blocks(1024, 1024))
        A0 = collect(DA)
        Dagger.spawn_datadeps() do
            for tile in DA.chunks
                Dagger.@spawn double!(InOut(tile))
            end
        end
        doubled = collect(DA) ≈ 2 .* A0
        println("doubled on GPU: ", doubled)
        @assert doubled
        println("inside a GPU task, a tile is a ", fetch(Dagger.@spawn typeof(DA.chunks[1, 1])))

        # (3) Library code built on Datadeps works too: DArray matmul, now on
        #     the GPU.
        DC = DA * DA
        host_A = collect(DA)
        product_correct = collect(DC) ≈ host_A * host_A
        println("GPU matmul correct: ", product_correct)
        @assert product_correct
    end
end
# source-end

function run_demo()
    DemoPresentation.present(run_gpu; title="03 Single GPU",
                             files=[@__FILE__], demo="03_gpu")
end
if abspath(PROGRAM_FILE) == @__FILE__
    run_demo()
end
