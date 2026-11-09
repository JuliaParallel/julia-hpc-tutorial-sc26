# Background renderer for the final exercise's Visualization panel.
# The worker calls this after a successful run; the code is never part of the
# student's template and is not echoed as console source.
module FinalVisuals

using UnicodePlots, Statistics

const TYPES = ("vorticity", "streamfunction", "velocity")

showplot(io, p) = (show(IOContext(io, :color => true), MIME"text/plain"(), p); println(io))

function render(omega::AbstractMatrix, psi::AbstractMatrix;
                types = TYPES, lid::Real = 1.0)
    size(omega) == size(psi) || error("omega and psi must have the same shape")
    nx, ny = size(omega)
    io = IOBuffer()
    for kind in types
        if kind == "vorticity"
            # Symmetric clip so the lid layer does not wash out the vortex.
            z = quantile(abs.(vec(omega[2:end-1, 2:end-1])), 0.9)
            showplot(io, heatmap(permutedims(omega);
                                 title = "vorticity ω  (clipped at ±$(round(z, digits = 1)))",
                                 xlabel = "x", ylabel = "y", xfact = 1 / (nx - 1),
                                 yfact = 1 / (ny - 1), zlim = (-z, z),
                                 colormap = :coolwarm, width = nx, height = ny ÷ 2))
        elseif kind == "streamfunction"
            showplot(io, contourplot(range(0, 1, length = nx), range(0, 1, length = ny),
                                     permutedims(psi);
                                     title = "streamfunction ψ (streamlines)", xlabel = "x",
                                     ylabel = "y", width = 50, height = 25, levels = 8,
                                     colormap = :viridis))
        elseif kind == "velocity"
            h = 1.0 / (nx - 1)
            u = zeros(nx, ny)
            for j in 2:ny-1, i in 2:nx-1
                u[i, j] = (psi[i, j+1] - psi[i, j-1]) / (2h)
            end
            u[:, end] .= lid
            showplot(io, lineplot(u[nx ÷ 2, :], range(0, 1, length = ny);
                                  title = "u(y) at x = 0.5", xlabel = "u", ylabel = "y",
                                  width = 50, height = 14, canvas = BlockCanvas))
        else
            error("Unknown visualization: $kind")
        end
    end
    k = argmin(psi)
    println(io, "primary vortex: psi = ", round(psi[k], digits = 3),
            " at x = ", round((k[1] - 1) / (nx - 1), digits = 2),
            ", y = ", round((k[2] - 1) / (ny - 1), digits = 2))
    return String(take!(io))
end

end
