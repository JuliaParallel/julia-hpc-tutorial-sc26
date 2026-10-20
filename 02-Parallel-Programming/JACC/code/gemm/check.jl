using BenchmarkTools
include("main.jl")
include("solution.jl")

Arows = 2_000
Acols = 2_000
Bcols = 2_000

A_h, B_h, C_h = Ex_gemm.init(Arows, Acols, Bcols)
@time Ex_gemm.gemm!(A_h, B_h, C_h)
@time Ex_gemm.gemm!(A_h, B_h, C_h)

println()

A_d = JACC.to_device(A_h)
B_d = JACC.to_device(B_h)
C_d = JACC.to_device(C_h)
@time gemm_jacc!(A_d, B_d, C_d)
@time gemm_jacc!(A_d, B_d, C_d)
