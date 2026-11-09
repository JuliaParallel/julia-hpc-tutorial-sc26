# Solution: Gray-Scott reaction-diffusion with ADI splitting and Cholesky.
# The two fields are substrate u and activator v on a periodic n x n grid.
# Each time step is two half steps; one direction is implicit and the other
# explicit. The implicit 1D operators are constant, SPD, and factored once.

using Dagger, LinearAlgebra, Random

n = 128
dt = 1.0
Du, Dv = 0.16, 0.08
F, k = 0.055, 0.062
nsteps = 3000

Random.seed!(42)
u = ones(n, n)
v = 0.01 .* rand(n, n)
for _ in 1:14
    cx, cy = rand(6:n-5, 2)
    v[cx-3:cx+3, cy-3:cy+3] .= 1.0
    u[cx-3:cx+3, cy-3:cy+3] .= 0.5
end

# 1D periodic Laplacian. L is symmetric negative semidefinite and the
# first/last rows wrap around.
L = zeros(n, n)
for i in 1:n
    L[i, i] = -2
    L[i, mod1(i - 1, n)] = 1
    L[i, mod1(i + 1, n)] = 1
end

# A_u and A_v are I - (dt/2) D L, hence SPD; factor once and reuse.
Fu = cholesky(Symmetric(Matrix{Float64}(I, n, n) - (dt / 2) * Du * L))
Fv = cholesky(Symmetric(Matrix{Float64}(I, n, n) - (dt / 2) * Dv * L))

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

println("u range: ", extrema(u), " v range: ", extrema(v))
