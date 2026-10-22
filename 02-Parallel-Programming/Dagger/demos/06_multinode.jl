# One CPU matrix product across at least two physical compute nodes.
# From a two-node allocation: bash frontier/multinode.sh cpu
const DEMO_USE_GPU = false
const DEMO_USE_MPI = true
const DEMO_MIN_NODES = 2
const DEMO_MIN_GPUS = 0
const DEMO_DEFAULT_GPUS = "0"
include(joinpath(@__DIR__, "matrix_product.jl"))
