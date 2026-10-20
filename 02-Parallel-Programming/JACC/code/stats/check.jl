using BenchmarkTools

include("main.jl")
include("solution.jl")

N = 1_000_000
const DFloat = JACC.default_float()

x_h = Ex_stats.init(N; dtype = DFloat)

mn::DFloat = 0.0
mx::DFloat = 0.0
av::DFloat = 0.0

println("min_max (host)")
mn, mx = Ex_stats.min_max(x_h)
@btime Ex_stats.min_max($x_h)
@show mn, mx
println("min_max_avg (host)")
mn, mx, av = Ex_stats.min_max_avg(x_h)
@btime Ex_stats.min_max_avg($x_h)
@show mn, mx, av

println()

x_d = JACC.to_device(x_h)
println("min_max (jacc)")
mn, mx = min_max_jacc(x_d)
@btime min_max_jacc($x_d)
@show mn, mx
println("min_max (jacc)")
mn, mx, av = min_max_avg_jacc(x_d)
@btime min_max_avg_jacc($x_d)
@show mn, mx, av
