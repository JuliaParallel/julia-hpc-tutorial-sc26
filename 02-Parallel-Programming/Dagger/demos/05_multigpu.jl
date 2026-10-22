# One matrix product, one Julia process, two GPUs on one node by default.
#   bash frontier/launch.sh 1 demos/05_multigpu.jl
#   bash frontier/launch.sh 1 demos/05_multigpu.jl --gpus=4
# Optional same-node MPI check, one GPU per rank:
#   bash frontier/launch.sh 4 demos/05_multigpu.jl --mpi
const DEMO_USE_GPU = true
const DEMO_USE_MPI = "--mpi" in ARGS
const DEMO_MIN_NODES = 1
const DEMO_MIN_GPUS = 1
const DEMO_DEFAULT_GPUS = DEMO_USE_MPI ? "all" : "2"
include(joinpath(@__DIR__, "matrix_product.jl"))
