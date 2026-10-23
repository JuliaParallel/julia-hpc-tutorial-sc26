# One MPI matrix product with two distinct GPUs per rank, on one or several nodes.
# Odo tutorial: two ranks on the existing node. bash frontier/multinode.sh gpu
const DEMO_SOURCE_FILE = @__FILE__
# source-begin
const DEMO_USE_GPU = true
const DEMO_USE_MPI = true
const DEMO_MIN_RANKS = 2
const DEMO_MIN_GPUS = 2
const DEMO_DEFAULT_GPUS = "2"
# source-end
include(joinpath(@__DIR__, "matrix_product.jl"))
