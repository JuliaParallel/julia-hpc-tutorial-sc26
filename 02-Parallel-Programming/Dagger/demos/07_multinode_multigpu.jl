# One matrix product across at least two nodes, with two GPUs per MPI rank.
# From a two-node allocation: bash frontier/multinode.sh gpu
const DEMO_USE_GPU = true
const DEMO_USE_MPI = true
const DEMO_MIN_NODES = 2
const DEMO_MIN_GPUS = 2
const DEMO_DEFAULT_GPUS = "2"
include(joinpath(@__DIR__, "matrix_product.jl"))
