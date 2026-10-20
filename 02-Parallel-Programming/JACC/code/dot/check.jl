using BenchmarkTools
include("main.jl")
include("solution.jl")

N = 100_000_000

x_h, y_h = Ex_dot.init(N)
@time res = Ex_dot.dot(x_h, y_h)
@time Ex_dot.dot(x_h, y_h)
@show res

println()

x_d, y_d = JACC.to_device.((x_h, y_h))
@time res = dot_jacc(x_d, y_d)
@time dot_jacc(x_d, y_d)
@show res
