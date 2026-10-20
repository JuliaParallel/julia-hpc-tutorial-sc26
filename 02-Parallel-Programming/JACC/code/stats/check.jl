
include("main.jl")
include("solution.jl")

N = 500_000_000

x_h = Ex_stats.init(N)

@time mn, mx = Ex_stats.min_max(x_h)
@time mn, mx = Ex_stats.min_max(x_h)
@show mn, mx

println()

x_d = JACC.to_device(x_h)
@time mn, mx = min_max_jacc(x_d)
@time mn, mx = min_max_jacc(x_d)
@show mn, mx
