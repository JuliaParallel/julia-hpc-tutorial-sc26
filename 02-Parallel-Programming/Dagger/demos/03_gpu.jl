# Demo 3: the SAME Datadeps code on an AMD GPU. Only the scope changes.
# Dagger moves the tiles to the GPU, and swaps BLAS -> rocBLAS for you.
#   julia --project -t 4 demos/03_gpu.jl
using Dagger, AMDGPU, LinearAlgebra

# source-begin
double!(X) = (X .*= 2; nothing)
# source-end
isdefined(@__MODULE__, :DemoPresentation) || include(joinpath(@__DIR__, "presentation.jl"))

function run_demo()
    DemoPresentation.present(; title="03 Single GPU", files=[@__FILE__], demo=splitext(basename(@__FILE__))[1]) do
        # source-begin
        gpu = Dagger.scope(rocm_gpu=1)

        Dagger.with_options(scope=gpu) do
            DA = DArray(rand(4096, 4096), Blocks(1024, 1024))
            A0 = collect(DA)
            Dagger.spawn_datadeps() do
                for tile in DA.chunks
                    Dagger.@spawn double!(InOut(tile))   # unchanged from demo 2
                end
            end
            doubled = collect(DA) ≈ 2 .* A0
            println("doubled on GPU: ", doubled)
            @assert doubled
            println("inside a GPU task, a tile is a ", fetch(Dagger.@spawn typeof(DA.chunks[1, 1])))

            # Library code built on Datadeps works too: DArray matmul, now on the GPU
            DC = DA * DA
            host_A = collect(DA)
            product_correct = collect(DC) ≈ host_A * host_A
            println("GPU matmul correct: ", product_correct)
            @assert product_correct
        end
        # source-end
    end
end
get(ENV, "SC26_DEMO_LOAD_ONLY", "0") == "1" || run_demo()
