# Rebuild after editing demo source: julia --project host/build_sources.jl
# Shipping the highlighted catalog keeps local Julia optional for app users.
include(joinpath(@__DIR__, "..", "demos", "presentation.jl"))
using .DemoPresentation, JSON3

root = dirname(@__DIR__)
catalog = map(sort(filter(f -> occursin(r"^0[1-7]_.*\.jl$", f), readdir(joinpath(root, "demos"))))) do file
    paths = [joinpath("demos", file)]
    startswith(file, r"0[5-7]") && push!(paths, joinpath("demos", "matrix_product.jl"))
    sources = map(paths) do path
        code = DemoPresentation.source_excerpt(joinpath(root, path))
        (; path, file=basename(path), code, ansi=DemoPresentation.highlighted(code))
    end
    (; demo=splitext(file)[1], sources)
end
open(io -> JSON3.write(io, catalog), joinpath(@__DIR__, "demo-sources.json"), "w")
println("Updated highlighted source catalog for $(length(catalog)) demos")
