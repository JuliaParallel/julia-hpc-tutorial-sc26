# Solution: 2D lid-driven cavity in streamfunction-vorticity form.
# This is the completed version of results/final-exercise/final_exercise.jl.

using Dagger, LinearAlgebra, Statistics, UnicodePlots

nx, ny = 32, 32
h = 1.0 / (nx - 1)
dt = 0.01
nu = 0.01
nsteps = 1000
lid = 1.0

omega = zeros(nx, ny)   # vorticity
psi = zeros(nx, ny)     # streamfunction
u = zeros(nx, ny)       # x velocity
v = zeros(nx, ny)       # y velocity

# 1. Build the 5-point Laplacian A = -d2/dx2 - d2/dy2 on the interior nodes.
#    psi = 0 on every wall, so the unknowns are the (nx-2) x (ny-2) interior.
#    Number them like vec(omega[2:end-1, 2:end-1]) (x varies fastest), so the
#    diagonal is 4/h^2 and every interior neighbour is -1/h^2.
A = zeros((nx - 2) * (ny - 2), (nx - 2) * (ny - 2))
for j in 1:ny-2, i in 1:nx-2
    k = i + (j - 1) * (nx - 2)
    A[k, k] = 4 / h^2
    i > 1 && (A[k, k - 1] = -1 / h^2)
    i < nx - 2 && (A[k, k + 1] = -1 / h^2)
    j > 1 && (A[k, k - (nx - 2)] = -1 / h^2)
    j < ny - 2 && (A[k, k + (nx - 2)] = -1 / h^2)
end

# 2. Factor A once with Cholesky. A is symmetric positive definite, and the
#    matrix never changes, so every time step can reuse F.
F = cholesky(Symmetric(A))

# 3. Time loop.
domega = zeros(nx, ny)
for step in 1:nsteps
    # Advect and diffuse omega explicitly in the interior. Use a copy (or a
    # separate domega array): the stencil reads the neighbouring cells.
    for j in 2:ny-1, i in 2:nx-1
        lap = (omega[i+1, j] + omega[i-1, j] + omega[i, j+1] + omega[i, j-1] -
               4omega[i, j]) / h^2
        du = u[i, j] > 0 ? (omega[i, j] - omega[i-1, j]) / h :
                           (omega[i+1, j] - omega[i, j]) / h
        dv = v[i, j] > 0 ? (omega[i, j] - omega[i, j-1]) / h :
                           (omega[i, j+1] - omega[i, j]) / h
        domega[i, j] = nu * lap - u[i, j] * du - v[i, j] * dv
    end
    omega[2:ny-1, 2:nx-1] .+= dt .* domega[2:ny-1, 2:nx-1]

    # Solve A psi = omega in the interior with the cached factor.
    psi[2:ny-1, 2:nx-1] .= reshape(F \ vec(omega[2:ny-1, 2:nx-1]), nx - 2, ny - 2)

    # u = d psi/dy, v = -d psi/dx, then update the wall vorticity (Thom):
    #   omega_wall = -2 psi_next_to_wall / h^2, minus 2 lid / h on the top wall.
    for j in 2:ny-1, i in 2:nx-1
        u[i, j] = (psi[i, j+1] - psi[i, j-1]) / (2h)
        v[i, j] = -(psi[i+1, j] - psi[i-1, j]) / (2h)
    end
    u[:, ny] .= lid
    omega[1, :] .= @views -2 .* psi[2, :] ./ h^2
    omega[end, :] .= @views -2 .* psi[end-1, :] ./ h^2
    omega[:, 1] .= @views -2 .* psi[:, 2] ./ h^2
    omega[:, end] .= @views -2 .* psi[:, end-1] ./ h^2 .- 2lid / h
end

# 4. Visuals (given -- no need to change).
showplot(p) = (show(IOContext(stdout, :color => !haskey(ENV, "NO_COLOR")),
                    MIME"text/plain"(), p); println())
z = quantile(abs.(vec(omega[2:end-1, 2:end-1])), 0.9)   # clip the wall layer
showplot(heatmap(permutedims(omega);
                 title = "vorticity ω  (clipped at ±$(round(z, digits = 1)))",
                 xlabel = "x", ylabel = "y", xfact = 1 / (nx - 1), yfact = 1 / (ny - 1),
                 zlim = (-z, z), colormap = :coolwarm, width = nx, height = ny ÷ 2))
showplot(contourplot(range(0, 1, length = nx), range(0, 1, length = ny), permutedims(psi);
                     title = "streamfunction ψ (streamlines)", xlabel = "x", ylabel = "y",
                     width = 50, height = 25, levels = 8, colormap = :viridis))
showplot(lineplot(u[nx ÷ 2, :], range(0, 1, length = ny);
                  title = "u(y) at x = 0.5", xlabel = "u", ylabel = "y",
                  width = 50, height = 14, canvas = BlockCanvas))

k = argmin(psi)
println("primary vortex: psi = ", round(psi[k], digits = 3),
        " at x = ", round((k[1] - 1) * h, digits = 2),
        ", y = ", round((k[2] - 1) * h, digits = 2))
