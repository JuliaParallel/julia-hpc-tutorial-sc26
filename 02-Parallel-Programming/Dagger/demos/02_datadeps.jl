# Demo 2: Datadeps. Tasks may mutate their arguments; you declare how
# (In / Out / InOut) and Dagger orders and parallelizes the tasks for you.
#
#   julia --project -t 4 demos/02_datadeps.jl
#   bash andromeda/launch.sh 1 0 demos/02_datadeps.jl

isdefined(@__MODULE__, :DemoPresentation) || include(joinpath(@__DIR__, "presentation.jl"))

# source-begin
using Dagger

# Small tile operations; the ! means they mutate their first argument.
add!(X, Y) = (X .+= Y; nothing)
double!(X) = (X .*= 2; nothing)

function run_datadeps()
    # (1) Order follows the data: `copyto!` reads B, so it waits for `add!`.
    A = rand(1000); B = rand(1000); C = zeros(1000)
    B0 = copy(B)
    Dagger.spawn_datadeps() do
        Dagger.@spawn add!(InOut(B), In(A))      # B += A
        Dagger.@spawn copyto!(Out(C), In(B))     # C = B
    end
    ordered = C ≈ B0 .+ A
    println("ordered correctly: ", ordered)
    @assert ordered
    # Delete every `Dagger.@spawn` and In/Out/InOut wrapper: this is still
    # correct serial code. Datadeps keeps the program's sequential meaning.

    # (2) Parallelism comes from independent data. A DArray is a grid of
    #     tiles: 16 tiles give 16 independent tasks.
    DA = DArray(rand(4096, 4096), Blocks(1024, 1024))
    A0 = collect(DA)
    Dagger.spawn_datadeps() do
        for tile in DA.chunks
            Dagger.@spawn double!(InOut(tile))
        end
    end
    doubled = collect(DA) ≈ 2 .* A0
    println("16 tiles doubled in parallel: ", doubled)
    @assert doubled
end
# source-end

function run_demo()
    DemoPresentation.present(run_datadeps; title="02 Mutable data and tiles",
                             files=[@__FILE__], demo="02_datadeps")
end
if abspath(PROGRAM_FILE) == @__FILE__
    run_demo()
end
