# Solution: Gray-Scott reaction-diffusion with ADI splitting and Cholesky.
# The two fields are substrate u and activator v on a periodic n x n grid.
# Each time step is two half steps; one direction is implicit and the other
# explicit. The implicit 1D operators are constant, SPD, and factored once.
#
# The console runs simulate! and then its Setup and Use editor; the Setup
# and Use below is the console's CPU, one-process version, so this file
# also runs on its own:   julia --project -t 4 solution/final_exercise.jl

using Dagger, LinearAlgebra

# Advance the substrate u and the activator v by nsteps ADI time steps,
# in place. p holds the parameters from Setup and Use: n, dt, Du, Dv, F, k.
function simulate!(u, v, p, nsteps)
    (; n, dt, Du, Dv, F, k) = p

    # 1. 1D periodic Laplacian. L is symmetric negative semidefinite and the
    #    first/last rows wrap around.
    L = zeros(n, n)
    for i in 1:n
        L[i, i] = -2
        L[i, mod1(i - 1, n)] = 1
        L[i, mod1(i + 1, n)] = 1
    end

    # 2. A_u and A_v are I - (dt/2) D L, hence SPD; factor once and reuse.
    Fu = cholesky(Symmetric(Matrix{Float64}(I, n, n) - (dt / 2) * Du * L))
    Fv = cholesky(Symmetric(Matrix{Float64}(I, n, n) - (dt / 2) * Dv * L))

    # 3. Time loop: two ADI half steps per time step.
    ru = similar(u)
    rv = similar(v)
    for step in 1:nsteps
        # First half step: x implicit, y explicit.
        mul!(ru, u, L)
        mul!(rv, v, L)
        @. ru = u + (dt / 2) * (Du * ru - u * v^2 + F * (1 - u))
        @. rv = v + (dt / 2) * (Dv * rv + u * v^2 - (F + k) * v)
        for j in 1:n
            u[:, j] = Fu \ ru[:, j]
            v[:, j] = Fv \ rv[:, j]
        end

        # Second half step: y implicit, x explicit.
        mul!(ru, L, u)
        mul!(rv, L, v)
        @. ru = u + (dt / 2) * (Du * ru - u * v^2 + F * (1 - u))
        @. rv = v + (dt / 2) * (Dv * rv + u * v^2 - (F + k) * v)
        for i in 1:n
            u[i, :] = Fu \ ru[i, :]
            v[i, :] = Fv \ rv[i, :]
        end
    end
    return u, v
end

# Setup and Use: where the tasks run, the parameters, the initial fields,
# the call to simulate!, and the check. Run final exercise runs the
# Simulation first, then this code.
# The Visualization panel then draws u and v.

using Dagger, LinearAlgebra, Random

# (1) Where the tasks run: the CPU threads.
scope = Dagger.scope(worker=1)

# (2) Parameters: grid size, time step, diffusion rates, feed and kill rates.
n, dt = 256, 1.0
Du, Dv = 0.16, 0.08
F, k = 0.055, 0.062
nsteps = 3600

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

# (4) Run the simulation where the scope says; it updates u and v in place.
Dagger.with_options(scope=scope) do
    simulate!(u, v, (; n, dt, Du, Dv, F, k), nsteps)
end

# (5) Check and report: both fields must stay finite.
all(isfinite, u) && all(isfinite, v) || error("the simulation blew up")
println("u range: ", extrema(u), " v range: ", extrema(v))
