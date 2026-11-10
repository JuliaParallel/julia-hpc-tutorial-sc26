# Solution: Gray-Scott reaction-diffusion with ADI splitting, stencils for the
# explicit terms, and Cholesky for the implicit ones. The fields u and v are
# DArrays split into panels of whole columns. A half step is explicit along y
# (a stencil), implicit along x (one Cholesky solve per panel), and ends with
# a transpose, so the next half step treats the other direction implicitly
# with the same code. Two half steps make one time step.
#
# The console runs simulate! and then its Setup and Use editor; the Setup
# and Use below is the console's CPU, one-process version, so this file
# also runs on its own:   julia --project -t 4 solution/final_exercise.jl

using Dagger, LinearAlgebra
import Dagger: @stencil, Wrap
import LinearAlgebra: LAPACK

# Second difference along y, the second index, of a 3 x 3 neighbourhood nb;
# nb[2, 2] is the point itself.
d2y(nb) = nb[2, 1] - 2nb[2, 2] + nb[2, 3]

# solve_columns!('L', C, B) solves A X = B for every column of the panel B, in
# place, from the lower Cholesky factor C of A. It is LAPACK.potrs! under
# another name, so Dagger swaps in cuSOLVER when the task runs on a GPU.
const solve_columns! = LAPACK.potrs!

# Rows `to` of the panel dst <- the transpose of rows `from` of the panel src.
function transpose_block!(dst, src, from, to)
    dst[to, :] .= transpose(view(src, from, :))
    return nothing
end

# One ADI half step, from (u, v) back into (u, v), transposed.
function half_step!(u, v, ru, rv, Cu, Cv, p)
    (; dt, Du, Dv, F, k) = p
    # (a) Explicit along y, plus the reaction: a stencil over every tile.
    @stencil begin
        ru[idx] = u[idx] + dt / 2 * (Du * d2y(@neighbors(u[idx], 1, Wrap())) -
                                     u[idx] * v[idx]^2 + F * (1 - u[idx]))
        rv[idx] = v[idx] + dt / 2 * (Dv * d2y(@neighbors(v[idx], 1, Wrap())) +
                                     u[idx] * v[idx]^2 - (F + k) * v[idx])
    end
    # (b) Implicit along x: each panel holds whole columns, so the solves are
    #     independent. (c) Then u = ru' and v = rv', one block at a time.
    panels = size(u.chunks, 2)
    width = size(u, 2) ÷ panels
    block(i) = (i - 1) * width + 1:i * width
    Dagger.spawn_datadeps() do
        for j in 1:panels
            Dagger.@spawn solve_columns!('L', In(Cu), InOut(ru.chunks[1, j]))
            Dagger.@spawn solve_columns!('L', In(Cv), InOut(rv.chunks[1, j]))
        end
        for j in 1:panels, i in 1:panels
            Dagger.@spawn transpose_block!(InOut(u.chunks[1, j]), In(ru.chunks[1, i]), block(j), block(i))
            Dagger.@spawn transpose_block!(InOut(v.chunks[1, j]), In(rv.chunks[1, i]), block(j), block(i))
        end
    end
    return nothing
end

# Advance the substrate u and the activator v by nsteps ADI time steps, in
# place. p holds the parameters from Setup and Use: n, dt, Du, Dv, F, k.
function simulate!(u, v, p, nsteps)
    (; n, dt, Du, Dv) = p

    # 1. 1D periodic Laplacian: diagonal -2, neighbours +1, wrapping around.
    L = zeros(n, n)
    for i in 1:n
        L[i, i] = -2
        L[i, mod1(i - 1, n)] = 1
        L[i, mod1(i + 1, n)] = 1
    end

    # 2. A = I - (dt/2) D L is SPD and constant: factor it once with Cholesky
    #    and keep the lower factor for every half step.
    Cu = Matrix(cholesky(Symmetric(I - (dt / 2) * Du * L)).L)
    Cv = Matrix(cholesky(Symmetric(I - (dt / 2) * Dv * L)).L)

    # 3. Time loop: x implicit, then (after the transpose) y implicit.
    ru, rv = similar(u), similar(v)
    for step in 1:nsteps
        half_step!(u, v, ru, rv, Cu, Cv, p)
        half_step!(u, v, ru, rv, Cu, Cv, p)
    end
    return u, v
end

# Setup and Use: where the tasks run, the parameters, the initial fields,
# the call to simulate!, and the check. Run final exercise runs the
# Simulation first, then this code.
# The Visualization panel then draws u and v.

using Dagger, LinearAlgebra, Random

BLAS.set_num_threads(1)   # Dagger runs one solve per thread

# (1) Where the tasks run: the CPU threads.
scope = Dagger.scope(worker=1)

# (2) Parameters: grid size, time step, diffusion rates, feed and kill rates.
n, dt = 256, 2.0
Du, Dv = 0.16, 0.08
F, k = 0.055, 0.062
nsteps = 1800

# (3) Initial fields: substrate u = 1 and a little activator v, plus 14
#     seeded squares where the reaction starts. The seed makes every MPI
#     rank start from the same fields.
Random.seed!(42)
u = ones(n, n)
v = 0.01 .* rand(n, n)
for _ in 1:14
    cx, cy = rand(6:n-5, 2)
    v[cx-3:cx+3, cy-3:cy+3] .= 1.0
    u[cx-3:cx+3, cy-3:cy+3] .= 0.5
end

# (4) Split u and v into two panels of whole columns, run the simulation
#     where the scope says, and gather the fields.
u, v = Dagger.with_options(scope=scope) do
    U = DArray(u, Blocks(n, n ÷ 2))
    V = DArray(v, Blocks(n, n ÷ 2))
    simulate!(U, V, (; n, dt, Du, Dv, F, k), nsteps)
    collect(U), collect(V)
end

# (5) Check and report: both fields must stay finite.
all(isfinite, u) && all(isfinite, v) || error("the simulation blew up")
println("u range: ", extrema(u), " v range: ", extrema(v))
