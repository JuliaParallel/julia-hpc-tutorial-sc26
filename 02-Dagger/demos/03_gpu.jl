# Demo 3: the SAME Datadeps code on an AMD GPU. Only the scope changes.
# Dagger moves the tiles to the GPU, and swaps BLAS -> rocBLAS for you.
#   julia --project -t 4 demos/03_gpu.jl
using Dagger, AMDGPU, LinearAlgebra

double!(X) = (X .*= 2; nothing)
gpu = Dagger.scope(rocm_gpu=1)

Dagger.with_options(scope=gpu) do
    DA = DArray(rand(4096, 4096), Blocks(1024, 1024))
    A0 = collect(DA)
    Dagger.spawn_datadeps() do
        for tile in DA.chunks
            Dagger.@spawn double!(InOut(tile))   # unchanged from demo 2
        end
    end
    println("doubled on GPU: ", collect(DA) ≈ 2 .* A0)
    println("inside a GPU task, a tile is a ", fetch(Dagger.@spawn typeof(DA.chunks[1, 1])))

    # Library code built on Datadeps works too: DArray matmul, now on the GPU
    DC = DA * DA
    println("GPU matmul correct: ", collect(DC) ≈ collect(DA) * collect(DA))
end
