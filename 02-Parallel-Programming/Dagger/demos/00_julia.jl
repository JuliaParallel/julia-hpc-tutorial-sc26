# Demo 0: the Julia features used throughout the Dagger sections.
#
#   julia --project -t 4 demos/00_julia.jl

isdefined(@__MODULE__, :DemoPresentation) || include(joinpath(@__DIR__, "presentation.jl"))

# source-begin
using Dagger, LinearAlgebra

# A function can mutate its argument; the trailing ! is only a convention.
scale!(A::AbstractMatrix, alpha) = (A .*= alpha; A)

# Multiple dispatch chooses the method from the argument types.
describe(x::Number) = "a number"
describe(x::AbstractArray) = "an array with $(length(x)) elements"

# A do block passes its function as the first argument. Dagger's
# `spawn_datadeps() do ... end` uses the same syntax.
with_column(f, A) = f(@view A[:, 1])

function tour_of_julia()
    A = reshape(collect(1.0:12.0), 3, 4)

    # (1) Dispatch: the same name, a different method per argument type.
    println("describe(A) = ", describe(A))
    println("describe(3.5) = ", describe(3.5))

    # (2) Broadcasting applies an operation elementwise; the dot before `=`
    #     mutates in place. This is how tile kernels update whole arrays.
    B = scale!(copy(A), 2)
    println("2 .* A = ", B)

    # (3) A view aliases its parent; copying a slice does not.
    col = @view A[:, 2]
    col .= 0
    println("after writing the view, A[:, 2] = ", A[:, 2])

    # (4) A do block, as in `Dagger.spawn_datadeps() do ... end`.
    column_sum = with_column(A) do column
        sum(column)
    end
    println("sum of column 1 via a do block = ", column_sum)

    # (5) A closure captures values from the surrounding scope.
    addn(n) = x -> x + n
    println("addn(10)(5) = ", addn(10)(5))

    # (6) A named tuple is convenient for task metadata.
    tile = (name = "tile-3", rows = 2:3)
    println("tile.name = ", tile.name, ", tile.rows = ", tile.rows)

    # (7) A for loop walks a range or any iterable; the tile demos use this
    #     shape to visit every tile of a DArray.
    range_sum = 0
    for i in 1:4
        range_sum += i
    end
    println("sum 1:4 = ", range_sum)

    # (8) Nested loops share one `for`; `enumerate` pairs an index with each
    #     item, as in `for (i, tile) in enumerate(DA.chunks)`.
    C = zeros(2, 3)
    for j in 1:3, i in 1:2
        C[i, j] = 10i + j
    end
    for (k, value) in enumerate([2.0, 3.0, 5.0])
        println("item ", k, " = ", value)
    end
    println("C = ", C)

    # (9) Macros transform code before it runs: `@views` makes slices views,
    #     `@.` turns a whole expression into broadcasting.
    @views begin
        firstrow = A[1, :]
        firstrow .= 1
    end
    @. B = B + 1
    println("after @views and @.: A[1, :] = ", A[1, :])
    println("B[1, 1] after @. = ", B[1, 1])

    # (10) A preview of Dagger: `@spawn` returns a task, `fetch` waits for it.
    task = Dagger.@spawn sum(A)
    println("first Dagger task result = ", fetch(task))
end
# source-end

function run_demo()
    DemoPresentation.present(tour_of_julia; title="00 Julia basics",
                             files=[@__FILE__], demo="00_julia")
end
if abspath(PROGRAM_FILE) == @__FILE__
    run_demo()
end
