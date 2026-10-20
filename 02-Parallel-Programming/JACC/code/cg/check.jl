include("main.jl")
include("solution.jl")

SIZE = 100000000
data = Ex_cg.init(SIZE; dtype = JACC.default_float())

data_jacc = init_jacc(data)
@time cg_jacc(data_jacc)
@time cg_jacc(data_jacc)

println()

data_multi = init_jacc_multi(data)
@time cg_jacc_multi(data_multi)
@time cg_jacc_multi(data_multi)

println()

data_async = init_jacc_async(data)
@time cg_jacc_async(data_async)
@time cg_jacc_async(data_async)
