# One CPU matrix product across MPI ranks, on one or several nodes.
# Odo tutorial: two ranks on the existing node. bash frontier/multinode.sh cpu
const DEMO_SOURCE_FILE = @__FILE__
# source-begin
const DEMO_USE_GPU = false
const DEMO_USE_MPI = true
const DEMO_MIN_RANKS = 2
const DEMO_MIN_GPUS = 0
const DEMO_DEFAULT_GPUS = "0"
# source-end
include(joinpath(@__DIR__, "matrix_product.jl"))
