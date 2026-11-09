# Field extraction for the final exercise's Visualization panel.
# The worker sends these arrays to the console, which draws true raster plots
# in the browser. No plotting package is needed on the compute node.
module FinalVisuals

using Base64

const TYPES = ("substrate", "activator")

rows(A, digits) = [[round(A[i, j], digits = digits) for j in 1:size(A, 2)]
                   for i in 1:size(A, 1)]

# Quantize a frame to 8 bits and base64-encode it in Julia's column-major
# order. The browser unpacker reads index i + j*nx for column i, row j.
function encode(A::AbstractMatrix, lo::Real, hi::Real)
    span = hi > lo ? hi - lo : 1.0
    bytes = Vector{UInt8}(undef, length(A))
    @inbounds for (k, value) in enumerate(A)
        bytes[k] = round(UInt8, clamp((value - lo) / span, 0.0, 1.0) * 255)
    end
    return base64encode(bytes)
end

function series_extrema(series, field)
    lo, hi = Inf, -Inf
    for frame in series
        a, b = extrema(getfield(frame, field))
        lo = min(lo, a)
        hi = max(hi, b)
    end
    return lo, hi
end

function payload(u::AbstractMatrix, v::AbstractMatrix;
                 frames = nothing, max_frames::Integer = 32, params = nothing)
    size(u) == size(v) || error("u and v must have the same shape")
    nx, ny = size(u)
    payload = (; nx, ny, u = rows(u, 4), v = rows(v, 4))
    params !== nothing && (payload = merge(payload, (; params)))
    if frames !== nothing && length(frames) > 1
        series = collect(frames)
        last = series[end]
        if !(last.u == u && last.v == v)
            push!(series, (; u, v))
        end
        count = min(max_frames, length(series))
        indices = unique(round.(Int, range(1, length(series), length = count)))
        umin, umax = series_extrema(series, :u)
        vmin, vmax = series_extrema(series, :v)
        encoded = [(; u = encode(series[i].u, umin, umax),
                      v = encode(series[i].v, vmin, vmax)) for i in indices]
        payload = merge(payload, (; frames = encoded,
                                    frame_range = (; u = (umin, umax),
                                                     v = (vmin, vmax))))
    end
    return payload
end

end
