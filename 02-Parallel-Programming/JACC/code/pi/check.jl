using BenchmarkTools
include("main.jl")
include("solution.jl")

N = 5_000_000_000

@time res = Ex_pi.est_pi(N)
@time Ex_pi.est_pi(N)
@show res

println()

@time res = est_pi_jacc(N)
@time est_pi_jacc(N)
@show res

println()

@time res = est_pi_jacc_2(N)
@time est_pi_jacc_2(N)
@show res
