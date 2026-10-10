# Demo 2: Datadeps. Tasks may mutate their arguments; you declare how
# (In / Out / InOut) and Dagger orders and parallelizes the tasks for you.
#   julia --project -t 4 demos/02_datadeps.jl
using Dagger

add!(X, Y) = (X .+= Y; nothing)
double!(X) = (X .*= 2; nothing)

A = rand(1000); B = rand(1000); C = zeros(1000)
B0 = copy(B)
Dagger.spawn_datadeps() do
    Dagger.@spawn add!(InOut(B), In(A))      # B += A
    Dagger.@spawn copyto!(Out(C), In(B))     # C = B  (waits for add! -- reads B)
end
println("ordered correctly: ", C ≈ B0 .+ A)
# Delete every `Dagger.@spawn` and the In/InOut wrappers: this is still
# correct serial code. Datadeps keeps the program's sequential meaning.

# Parallelism comes from independent data: a DArray is a grid of tiles
DA = DArray(rand(4096, 4096), Blocks(1024, 1024))
A0 = collect(DA)
Dagger.spawn_datadeps() do
    for tile in DA.chunks                    # 16 tiles -> 16 independent tasks
        Dagger.@spawn double!(InOut(tile))
    end
end
println("16 tiles doubled in parallel: ", collect(DA) ≈ 2 .* A0)
