# Frames of the final exercise's Gray-Scott run for the slides: the activator
# v at three times, with the console's parameters and initial fields. Serial
# ADI with Cholesky, standard library only:
#   julia presentations/dagger/gray_scott_data.jl
# writes data/gray-scott-v.bin: a 3 x 256 x 256 UInt8 array (column-major),
# each frame scaled to its own range. build_figures.py turns it into a PNG.
using LinearAlgebra, Random

function frames(times; n=256, dt=2.0, Du=0.16, Dv=0.08, F=0.055, k=0.062)
    Random.seed!(42)
    u = ones(n, n)
    v = 0.01 .* rand(n, n)
    for _ in 1:14
        cx, cy = rand(6:n-5, 2)
        v[cx-3:cx+3, cy-3:cy+3] .= 1.0
        u[cx-3:cx+3, cy-3:cy+3] .= 0.5
    end
    L = zeros(n, n)
    for i in 1:n
        L[i, i] = -2
        L[i, mod1(i - 1, n)] = 1
        L[i, mod1(i + 1, n)] = 1
    end
    Fu = cholesky(Symmetric(I - (dt / 2) * Du * L))
    Fv = cholesky(Symmetric(I - (dt / 2) * Dv * L))
    ru, rv = similar(u), similar(v)
    out = UInt8[]
    step = 0
    for t in times
        while step * dt < t
            for direction in 1:2
                mul!(ru, u, L); mul!(rv, v, L)   # explicit along the second index
                @. ru = u + (dt / 2) * (Du * ru - u * v^2 + F * (1 - u))
                @. rv = v + (dt / 2) * (Dv * rv + u * v^2 - (F + k) * v)
                # implicit along the first index, then swap the directions
                u .= permutedims(Fu \ ru)
                v .= permutedims(Fv \ rv)
            end
            step += 1
        end
        lo, hi = extrema(v)
        append!(out, round.(UInt8, clamp.((v .- lo) ./ (hi - lo), 0, 1) .* 255))
    end
    return out
end

data = frames((200, 1000, 6000))
write(joinpath(@__DIR__, "data", "gray-scott-v.bin"), data)
println("wrote ", length(data), " bytes")
