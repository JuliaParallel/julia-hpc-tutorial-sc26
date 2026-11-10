# Heat diffusion with Dagger stencils -- YOUR FILE TO EDIT.
#
# u is the temperature of a periodic plate, a DArray split into tiles. One
# explicit time step of the heat equation is
#     u <- u + dt * D * laplacian(u)
# where the 5-point Laplacian of a point reads its four nearest neighbours.
# exercise/serial_heat.jl does exactly this with plain loops. Make it Dagger:
#   1. write laplacian(nb) for a 3 x 3 neighbourhood nb,
#   2. make each time step one `@stencil` block of two statements:
#      scratch gets the new temperature, read from
#      `@neighbors(u[idx], 1, Wrap())`, then u gets scratch back.
# Dagger runs one task per tile and copies the neighbouring tiles' edges.

using Dagger
import Dagger: @stencil, Wrap

# The 5-point Laplacian of a 3 x 3 neighbourhood nb. nb[2, 2] is the point
# itself; nb[1, 2] and nb[3, 2] are its neighbours along x (the first
# index), nb[2, 1] and nb[2, 3] its neighbours along y (the second index).
laplacian(nb) = error("write the 5-point Laplacian of nb")

# Advance u by nsteps explicit steps, in place. scratch has u's tiles.
function my_heat!(u, scratch, D, dt, nsteps)
    for step in 1:nsteps
        # One @stencil block: scratch from u's neighbourhood, then u from scratch.
    end
    return u
end
