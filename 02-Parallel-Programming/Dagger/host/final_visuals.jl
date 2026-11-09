# Field extraction for the final exercise's Visualization panel.
# The worker sends these arrays to the console, which draws true raster plots
# in the browser. No plotting package is needed on the compute node.
module FinalVisuals

const TYPES = ("vorticity", "streamfunction")

rows(A, digits) = [[round(A[i, j], digits = digits) for j in 1:size(A, 2)]
                   for i in 1:size(A, 1)]

function payload(omega::AbstractMatrix, psi::AbstractMatrix;
                 frames = nothing, max_frames::Integer = 16)
    size(omega) == size(psi) || error("omega and psi must have the same shape")
    nx, ny = size(omega)
    h = 1.0 / (nx - 1)
    k = argmin(psi)
    payload = (; nx, ny, omega = rows(omega, 4), psi = rows(psi, 5),
                 vortex = (; psi = round(psi[k], digits = 4),
                             x = round((k[1] - 1) * h, digits = 3),
                             y = round((k[2] - 1) * h, digits = 3)))
    if frames !== nothing && length(frames) > 1
        series = collect(frames)
        last = series[end]
        if !(last.omega == omega && last.psi == psi)
            push!(series, (; omega, psi))
        end
        count = min(max_frames, length(series))
        indices = unique(round.(Int, range(1, length(series), length = count)))
        payload = merge(payload, (; frames = [(; omega = rows(series[i].omega, 4),
                                                 psi = rows(series[i].psi, 5))
                                              for i in indices]))
    end
    return payload
end

end
