# INSTRUCTOR SOLUTION -- Gray-Scott reaction-diffusion with ADI splitting,
# Cholesky line solves with your my_cholesky!, and DArray panels.
#
# The two fields are substrate u and activator v on a periodic n x n grid.
# A time step is two ADI half steps. A half step is explicit in one direction
# and implicit in the other; the implicit part is one small SPD system per
# grid line, all with the same matrix, factored once with my_cholesky!.
#
# u and v are DArrays split into panels of whole columns, spread over the
# processors and MPI ranks. Each half step runs one task per panel that
# solves all its lines (it writes its own panel: InOut, and reads its
# neighbours: In), then one task per panel that stores the result
# transposed, so the next half step solves along the other direction with
# the same code. Datadeps runs each task where the panel it writes lives,
# on CPU threads, GPUs or MPI ranks, wherever the scope says.
#
# The console runs your my_cholesky!, then simulate!, then its Setup and Use;
# the Setup and Use below is the console's CPU, one-process version, so this
# file also runs on its own:   julia --project -t 4 solution/final_exercise.jl

using Dagger, LinearAlgebra

# my_cholesky! is your function from the Cholesky exercise: the console runs it
# before this code. On its own, this file uses the reference one.
isdefined(@__MODULE__, :my_cholesky!) || include(joinpath(@__DIR__, "my_cholesky.jl"))

# The lower Cholesky factor C of A (A = C * C'), with your my_cholesky!: tile A
# into bs x bs blocks, factor the tiles in place, and gather the lower triangle.
function lower_factor(A, bs)
    DA = DArray(A, Blocks(bs, bs))
    my_cholesky!(DA.chunks)
    return Matrix(LowerTriangular(collect(DA)))
end

# Solve A X = R for every column of R, in place, from the lower Cholesky
# factor C of A = C * C': two triangular solves (they work on GPUs too).
function solve_lines!(R, C)
    ldiv!(LowerTriangular(C), R)
    ldiv!(LowerTriangular(C)', R)
    return nothing
end

# Second difference across the columns of the panel x. The panels before and
# after it (wrapping around) supply the neighbours of its edge columns.
across(x, before, after) =
    hcat(before[:, end:end], x[:, 1:end-1]) .- 2 .* x .+ hcat(x[:, 2:end], after[:, 1:1])

# One half step of both fields on one panel: explicit across the columns, with
# the reaction, then implicit along them (the line solves).
function half_step_panel!(su, sv, u, u_before, u_after, v, v_before, v_after, Cu, Cv, p)
    (; dt, Du, Dv, F, k) = p
    d2u = across(u, u_before, u_after)
    d2v = across(v, v_before, v_after)
    @. su = u + (dt / 2) * (Du * d2u - u * v^2 + F * (1 - u))
    @. sv = v + (dt / 2) * (Dv * d2v + u * v^2 - (F + k) * v)
    solve_lines!(su, Cu)
    solve_lines!(sv, Cv)
end

# Panel j of the transposed fields: its rows cols[i] are the transpose of rows
# cols[j] of solved panel i. `solved` holds the panels of su, then those of sv.
function transpose_panel!(u, v, cols, j, solved...)
    panels = length(cols)
    for i in 1:panels
        u[cols[i], :] .= permutedims(solved[i][cols[j], :])
        v[cols[i], :] .= permutedims(solved[panels + i][cols[j], :])
    end
    return nothing
end

# Advance the substrate u and the activator v by nsteps ADI time steps, in
# place. u and v are DArrays split into panels of whole columns; p holds the
# parameters from Setup and Use: n, dt, Du, Dv, F, k.
function simulate!(u, v, p, nsteps)
    (; n, dt, Du, Dv) = p

    # 1. 1D periodic Laplacian. L is symmetric negative semidefinite and the
    #    first/last rows wrap around.
    L = zeros(n, n)
    for i in 1:n
        L[i, i] = -2
        L[i, mod1(i - 1, n)] = 1
        L[i, mod1(i + 1, n)] = 1
    end

    # 2. A_u and A_v are I - (dt/2) D L, hence SPD and constant: factor each
    #    once with your my_cholesky!, in tiles of 128.
    Cu = lower_factor(I - (dt / 2) * Du * L, 128)
    Cv = lower_factor(I - (dt / 2) * Dv * L, 128)

    # 3. The panels: the columns of each, and DArrays for the solved panels,
    #    laid out like u and v.
    panels = length(u.chunks)
    cols = Tuple(Dagger.domainchunks(u)[q].indexes[2] for q in 1:panels)   # a tuple stays on the host
    before(q) = mod1(q - 1, panels)
    after(q) = mod1(q + 1, panels)
    su, sv = similar(u), similar(v)

    # 4. Time loop: two ADI half steps per time step, in one Datadeps region.
    for step in 1:nsteps
        Dagger.spawn_datadeps() do
            for half in 1:2
                # (a) Every panel's line solves, in parallel: each task reads
                #     its panel and the two next to it, and writes its own.
                for q in 1:panels
                    Dagger.@spawn half_step_panel!(InOut(su.chunks[q]), InOut(sv.chunks[q]),
                        In(u.chunks[q]), In(u.chunks[before(q)]), In(u.chunks[after(q)]),
                        In(v.chunks[q]), In(v.chunks[before(q)]), In(v.chunks[after(q)]),
                        In(Cu), In(Cv), p)
                end
                # (b) Store the result transposed, so the next half step solves
                #     along the other direction with the same code.
                for j in 1:panels
                    Dagger.@spawn transpose_panel!(InOut(u.chunks[j]), InOut(v.chunks[j]), cols, j,
                                                   map(In, su.chunks)..., map(In, sv.chunks)...)
                end
            end
        end
    end
    return u, v
end

# Setup and Use: where the tasks run, the parameters, the initial fields,
# the call to simulate!, and the check. Run final exercise runs the
# Simulation first, then this code.
# The Visualization panel then draws u and v.

using Dagger, LinearAlgebra, Random

BLAS.set_num_threads(1)   # Dagger runs one panel of line solves per thread

# (1) Where the tasks run: the CPU threads.
scope = Dagger.scope(worker=1)

# (2) Parameters: grid size, time step, diffusion rates, feed and kill rates.
n, dt = 512, 2.0
Du, Dv = 0.16, 0.08
F, k = 0.055, 0.062
nsteps = 1800

# Panels: 8 of 64 lines, about one per CPU thread (a one-process run
# has 7).
panels = 8

# (3) Initial fields: substrate u = 1 and a little activator v, plus 56
#     seeded squares where the reaction starts. The seed makes every MPI
#     rank start from the same fields.
Random.seed!(42)
u = ones(n, n)
v = 0.01 .* rand(n, n)
for _ in 1:56
    cx, cy = rand(6:n-5, 2)
    v[cx-3:cx+3, cy-3:cy+3] .= 1.0
    u[cx-3:cx+3, cy-3:cy+3] .= 0.5
end

# (4) Make u and v DArrays of `panels` blocks of whole columns, in CPU
#     memory: zeros spreads the blocks over this process, copyto! fills them.
home = Dagger.scope(worker=1)
U, V = Dagger.with_options(scope=home) do
    U = zeros(Blocks(n, cld(n, panels)), n, n)
    V = zeros(Blocks(n, cld(n, panels)), n, n)
    copyto!(U, u)
    copyto!(V, v)
    U, V
end

# (5) Run the simulation where the scope says; each task runs where the
#     panel it writes lives. Then gather the fields.
Dagger.with_options(scope=scope) do
    simulate!(U, V, (; n, dt, Du, Dv, F, k), nsteps)
end
u, v = collect(U), collect(V)

# (6) Check and report: both fields must stay finite.
all(isfinite, u) && all(isfinite, v) || error("the simulation blew up")
println("u range: ", extrema(u), " v range: ", extrema(v))
