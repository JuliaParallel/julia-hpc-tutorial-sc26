using BenchmarkTools
include("main.jl")
include("solution.jl")

N = 200_000_000

x_h, y_h = Ex_axpy.init(N)
@time Ex_axpy.axpy!(x_h, y_h)
@time Ex_axpy.axpy!(x_h, y_h)

println()

x_d, y_d = JACC.to_device.((x_h, y_h))
@time axpy_jacc!(x_d, y_d)
@time axpy_jacc!(x_d, y_d)
