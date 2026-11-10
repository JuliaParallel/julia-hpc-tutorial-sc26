# Serial heat diffusion on a periodic plate -- the STARTING POINT for the
# stencil exercise. Ordinary Julia, no Dagger: every point of the plate reads
# its four nearest neighbours, wrapping around at the edges.
#
# Your job (in my_stencil.jl) is to write the same update with Dagger's
# @stencil, so that every tile of a DArray is updated by its own task.

# The 5-point Laplacian of a 3 x 3 neighbourhood nb (nb[2, 2] is the point).
laplacian(nb) = nb[1, 2] + nb[3, 2] + nb[2, 1] + nb[2, 3] - 4nb[2, 2]

function serial_heat!(u, scratch, D, dt, nsteps)
    nx, ny = size(u)
    for step in 1:nsteps
        for j in 1:ny, i in 1:nx
            # The 3 x 3 neighbourhood of (i, j), wrapping around the edges.
            nb = [u[mod1(i + a, nx), mod1(j + b, ny)] for a in -1:1, b in -1:1]
            scratch[i, j] = u[i, j] + dt * D * laplacian(nb)
        end
        u .= scratch
    end
    return u
end

# Quick self-check: `julia --project exercise/serial_heat.jl`
if abspath(PROGRAM_FILE) == @__FILE__
    u = zeros(32, 32); u[10:14, 20:24] .= 1
    heat = sum(u)
    serial_heat!(u, similar(u), 0.2, 1.0, 50)
    println("serial_heat! keeps the total heat: ", sum(u) ≈ heat)
end
