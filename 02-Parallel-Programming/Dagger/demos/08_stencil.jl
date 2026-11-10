# Demo 8: stencils. `@stencil` updates every element of a DArray from its
# neighbours. Dagger runs one task per tile and copies the edges (halos) of
# neighbouring tiles for you. The same stencil runs on a GPU.
#
#   julia --project -t 4 demos/08_stencil.jl
#   bash andromeda/launch.sh 1 1 demos/08_stencil.jl

isdefined(@__MODULE__, :DemoPresentation) || include(joinpath(@__DIR__, "presentation.jl"))

# source-begin
using Dagger, CUDA, Random
import Dagger: @stencil, Wrap

function run_stencil()
    # (1) A noisy 1024 x 1024 image, tiled 4 x 4. Each tile is one task.
    image = rand(Xoshiro(1), 1024, 1024)
    A = DArray(copy(image), Blocks(256, 256))
    B = zeros(Blocks(256, 256), 1024, 1024)

    # (2) Every pixel becomes the mean of its 3 x 3 neighbourhood. A pixel on
    #     a tile edge reads its neighbours from the next tile; Wrap() makes the
    #     image periodic, so the outer edges wrap around.
    @stencil B[idx] = sum(@neighbors(A[idx], 1, Wrap())) / 9
    box = sum(circshift(image, (i, j)) for i in -1:1, j in -1:1) / 9
    println("3x3 mean matches circshift: ", collect(B) ≈ box)

    # (3) Index the neighbourhood: nb[2, 2] is the pixel, nb[2, 1] and
    #     nb[2, 3] its neighbours along the second index.
    @stencil B[idx] = begin
        nb = @neighbors(A[idx], 1, Wrap())
        (nb[2, 3] - nb[2, 1]) / 2
    end
    slope = (circshift(image, (0, -1)) - circshift(image, (0, 1))) / 2
    println("centred difference matches: ", collect(B) ≈ slope)

    # (4) The statements of a block run in order, each over every tile: smooth
    #     into B, then copy B back into A. Ten passes blur the noise away.
    for _ in 1:10
        @stencil begin
            B[idx] = sum(@neighbors(A[idx], 1, Wrap())) / 9
            A[idx] = B[idx]
        end
    end
    println("pixel range: ", extrema(image), " before, ", extrema(collect(A)), " after 10 passes")

    # (5) The same stencil on the GPU: only the scope changes.
    Dagger.with_options(scope=Dagger.scope(cuda_gpu=1)) do
        G = DArray(copy(image), Blocks(256, 256))
        H = zeros(Blocks(256, 256), 1024, 1024)
        @stencil H[idx] = sum(@neighbors(G[idx], 1, Wrap())) / 9
        println("3x3 mean on the GPU matches: ", collect(H) ≈ box)
        println("inside a GPU task, a tile is a ", fetch(Dagger.@spawn typeof(H.chunks[1, 1])))
    end
end
# source-end

function run_demo()
    DemoPresentation.present(run_stencil; title="08 Stencils",
                             files=[@__FILE__], demo="08_stencil")
end
if abspath(PROGRAM_FILE) == @__FILE__
    run_demo()
end
