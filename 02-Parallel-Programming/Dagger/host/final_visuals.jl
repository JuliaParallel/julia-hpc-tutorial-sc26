# Field extraction for the final exercise's Visualization panel.
# The worker sends these arrays to the console, which draws true raster plots
# in the browser. No plotting package is needed on the compute node.
module FinalVisuals

const TYPES = ("vorticity", "streamfunction", "velocity")

function payload(omega::AbstractMatrix, psi::AbstractMatrix)
    size(omega) == size(psi) || error("omega and psi must have the same shape")
    nx, ny = size(omega)
    h = 1.0 / (nx - 1)
    k = argmin(psi)
    return (; nx, ny,
              omega = [[round(omega[i, j], digits = 4) for j in 1:ny] for i in 1:nx],
              psi = [[round(psi[i, j], digits = 5) for j in 1:ny] for i in 1:nx],
              vortex = (; psi = round(psi[k], digits = 4),
                          x = round((k[1] - 1) * h, digits = 3),
                          y = round((k[2] - 1) * h, digits = 3)))
end

end
