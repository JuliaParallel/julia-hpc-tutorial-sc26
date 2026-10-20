using BenchmarkTools

include("main.jl")
include("solution.jl")

n_inputs = 256
n_x = 1024
n_y = 1024
n_filter = 5

out_h, in_h, filter_h = Ex_conv.init(
    n_inputs, n_x, n_y, n_filter; dtype = JACC.default_float())

# @time Ex_conv.conv!(out_h, in_h, filter_h)
# @time Ex_conv.conv!(out_h, in_h, filter_h)

println()

out, in, filter = JACC.to_device.((out_h, in_h, filter_h))

@btime conv_jacc!(out, in, filter)
# @time conv_jacc!(out, in, filter)

println()

@btime conv_jacc_shared!(out, in, filter)
# @time conv_jacc_shared!(out, in, filter)
