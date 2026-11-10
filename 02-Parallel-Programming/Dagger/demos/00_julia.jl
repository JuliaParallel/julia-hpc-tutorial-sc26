# Demo 0: the Julia features used throughout the Dagger sections.
#   julia --project -t 4 demos/00_julia.jl
# source-begin
# Setup: packages used by the examples.
using Dagger, LinearAlgebra
# source-end
isdefined(@__MODULE__, :DemoPresentation) || include(joinpath(@__DIR__, "presentation.jl"))

# source-begin
# Algorithm: helpers used by the examples below.
# A function can mutate its argument; the trailing ! is only a convention.
scale!(A::AbstractMatrix, alpha) = (A .*= alpha; A)

# Multiple dispatch chooses the method from the argument types.
describe(x::Number) = "a number"
describe(x::AbstractArray) = "an array with $(length(x)) elements"

# A do block passes its function as the first argument. Dagger's
# spawn_datadeps() do ... end uses the same Julia syntax.
with_column(f, A) = f(@view A[:, 1])
# source-end

function run_demo()
    DemoPresentation.present(; title="00 Julia basics", files=[@__FILE__], demo=splitext(basename(@__FILE__))[1]) do
        # source-begin
        # Setup and use: build a small array, then exercise each feature.
        A = reshape(collect(1.0:12.0), 3, 4)
        println("size(A) = ", size(A), ", describe(A) = ", describe(A))
        println("describe(3.5) = ", describe(3.5))

        # Broadcasting applies an operation elementwise; the dot before =
        # mutates in place. This is how tile kernels update whole arrays.
        B = scale!(copy(A), 2)
        println("2 .* A = ", B)

        # A view aliases its parent; copying a slice does not.
        col = @view A[:, 2]
        col .= 0
        println("after writing the view, A[:, 2] = ", A[:, 2])

        # do-block syntax, as in Dagger.spawn_datadeps() do ... end
        total = with_column(A) do column
            sum(column)
        end
        println("sum of column 1 via a do block = ", total)

        # Closures capture values from the surrounding scope.
        addn(n) = x -> x + n
        println("addn(10)(5) = ", addn(10)(5))

        # Named tuples are convenient for task metadata.
        tile = (name = "tile-3", rows = 2:3)
        println("tile.name = ", tile.name, ", tile.rows = ", tile.rows)

        # A for loop walks a range or any iterable. The tile demos use
        # this shape to visit every tile in a DArray.
        total = 0
        for i in 1:4
            total += i
        end
        println("sum 1:4 = ", total)

        # Nested loops share one `for`; `enumerate` pairs an index with
        # each item, as in `for (i, tile) in enumerate(DA.chunks)`.
        C = zeros(2, 3)
        for j in 1:3, i in 1:2
            C[i, j] = 10i + j
        end
        for (k, value) in enumerate([2.0, 3.0, 5.0])
            println("item ", k, " = ", value)
        end
        println("C = ", C)

        # Macros transform code before it runs. @views makes slices views;
        # @. turns a whole expression into broadcasting.
        @views begin
            firstrow = A[1, :]
            firstrow .= 1
        end
        @. B = B + 1
        println("after @views and @.: A[1, :] = ", A[1, :])
        println("B[1, 1] after @. = ", B[1, 1])

        # A Dagger macro preview: @spawn returns a task, fetch waits for it.
        task = Dagger.@spawn sum(A)
        println("first Dagger task result = ", fetch(task))
        # source-end
    end
end
get(ENV, "SC26_DEMO_LOAD_ONLY", "0") == "1" || run_demo()
