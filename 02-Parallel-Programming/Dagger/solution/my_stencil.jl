# INSTRUCTOR SOLUTION -- heat diffusion with Dagger stencils.
# Compare with exercise/serial_heat.jl: the update is the same; the loops over
# points became one @stencil statement, and Dagger runs one task per tile.

using Dagger
import Dagger: @stencil, Wrap

# The 5-point Laplacian of a 3 x 3 neighbourhood nb (nb[2, 2] is the point).
laplacian(nb) = nb[1, 2] + nb[3, 2] + nb[2, 1] + nb[2, 3] - 4nb[2, 2]

# Advance u by nsteps explicit steps, in place. scratch has u's tiles.
function my_heat!(u, scratch, D, dt, nsteps)
    for step in 1:nsteps
        # A statement must not read the neighbours of the array it writes, so
        # the new values go to scratch first and are copied back after.
        @stencil begin
            scratch[idx] = u[idx] + dt * D * laplacian(@neighbors(u[idx], 1, Wrap()))
            u[idx] = scratch[idx]
        end
    end
    return u
end
