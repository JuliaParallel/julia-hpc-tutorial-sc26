# INSTRUCTOR SOLUTION -- Gray-Scott reaction-diffusion with ADI splitting,
# Cholesky line solves, and Dagger Datadeps.
#
# The two fields are substrate u and activator v on a periodic n x n grid.
# A time step is two ADI half steps. A half step is explicit in one direction
# and implicit in the other; the implicit part is one small SPD system per
# grid line, all with the same matrix, factored once with Cholesky.
#
# The parallel work is the line solves. The columns are split into panels,
# and one Dagger task per panel and per field builds the right-hand sides of
# its lines and solves them all with the Cholesky factor. Tasks only read u
# and v (In) and each writes its own panel (InOut), so Dagger runs them
# together, on CPU threads, GPUs or MPI ranks, wherever the scope says.
#
# The x and y directions use the same operator, so both half steps run the
# same code: lines along the first index are solved, and the result is stored
# transposed. The second half step then solves along the other direction, and
# after two half steps the fields are the right way round again.
#
# The console runs simulate! and then its Setup and Use editor; the Setup
# and Use below is the console's CPU, one-process version, so this file
# also runs on its own:   julia --project -t 4 solution/final_exercise.jl

using Dagger, LinearAlgebra

# Solve A X = R for every column of R, in place, from the lower Cholesky
# factor C of A = C * C'. Two triangular solves; ldiv! also works on GPU arrays.
function solve_lines!(R, C)
    ldiv!(LowerTriangular(C), R)
    ldiv!(LowerTriangular(C)', R)
    return nothing
end

# One half step of the substrate u on one panel of columns `cols`; `left` and
# `right` are the columns before and after each of them, wrapping around.
# Explicit across the columns, with the reaction, then implicit along them.
function substrate_panel!(out, u, v, C, cols, left, right, p)
    (; dt, Du, F) = p
    uc, vc = view(u, :, cols), view(v, :, cols)
    @. out = uc + (dt / 2) * (Du * (u[:, left] - 2uc + u[:, right]) - uc * vc^2 + F * (1 - uc))
    solve_lines!(out, C)
end

# The same for the activator v.
function activator_panel!(out, u, v, C, cols, left, right, p)
    (; dt, Dv, F, k) = p
    uc, vc = view(u, :, cols), view(v, :, cols)
    @. out = vc + (dt / 2) * (Dv * (v[:, left] - 2vc + v[:, right]) + uc * vc^2 - (F + k) * vc)
    solve_lines!(out, C)
end

# Advance the substrate u and the activator v by nsteps ADI time steps,
# in place. p holds the parameters from Setup and Use:
# n, dt, Du, Dv, F, k and panels.
function simulate!(u, v, p, nsteps)
    (; n, dt, Du, Dv, panels) = p
    n % panels == 0 || error("panels must divide n")

    # 1. 1D periodic Laplacian. L is symmetric negative semidefinite and the
    #    first/last rows wrap around.
    L = zeros(n, n)
    for i in 1:n
        L[i, i] = -2
        L[i, mod1(i - 1, n)] = 1
        L[i, mod1(i + 1, n)] = 1
    end

    # 2. A_u and A_v are I - (dt/2) D L, hence SPD; factor once and keep the
    #    lower factors for every half step.
    Cu = Matrix(cholesky(Symmetric(I - (dt / 2) * Du * L)).L)
    Cv = Matrix(cholesky(Symmetric(I - (dt / 2) * Dv * L)).L)

    # The panels: their columns, each column's neighbours, and one output
    # array per panel and field.
    width = n ÷ panels
    cols = [(q - 1) * width + 1:q * width for q in 1:panels]
    left = [mod1.(c .- 1, n) for c in cols]
    right = [mod1.(c .+ 1, n) for c in cols]
    new_u = [zeros(n, width) for _ in 1:panels]
    new_v = [zeros(n, width) for _ in 1:panels]

    # 3. Time loop: two ADI half steps per time step.
    for step in 1:nsteps
        for half in 1:2
            # One task per panel and field: independent, so they run together.
            Dagger.spawn_datadeps() do
                for q in 1:panels
                    Dagger.@spawn substrate_panel!(InOut(new_u[q]), In(u), In(v), In(Cu),
                                                   cols[q], left[q], right[q], p)
                    Dagger.@spawn activator_panel!(InOut(new_v[q]), In(u), In(v), In(Cv),
                                                   cols[q], left[q], right[q], p)
                end
            end
            # Store the panels transposed: the next half step treats the
            # other direction implicitly.
            for q in 1:panels
                u[cols[q], :] .= transpose(new_u[q])
                v[cols[q], :] .= transpose(new_v[q])
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
n, dt = 512, 1.0
Du, Dv = 0.16, 0.08
F, k = 0.055, 0.062
nsteps = 3600

# Panels: one task per panel and field; 8 panels of 64 lines give each
# CPU thread one panel of each field (a one-process run has 7 threads).
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

# (4) Run the simulation where the scope says; it updates u and v in place.
Dagger.with_options(scope=scope) do
    simulate!(u, v, (; n, dt, Du, Dv, F, k, panels), nsteps)
end

# (5) Check and report: both fields must stay finite.
all(isfinite, u) && all(isfinite, v) || error("the simulation blew up")
println("u range: ", extrema(u), " v range: ", extrema(v))
