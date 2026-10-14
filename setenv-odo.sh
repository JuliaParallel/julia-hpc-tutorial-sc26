#!/bin/bash

# Set $SCRATCH environment variable to run Tutorial contents on the Odo OLCF system
# Odo is 32 nodes of Frontier, each node has 8 MI250X GPUs
export SCRATCH=/gpfs/wolf2/olcf/trn047/scratch/$USER

# Add pre-installed Julia v1.13.1 to the PATH
WORLDWORK=/gpfs/wolf2/olcf/trn047/world-shared
export PATH=$WORLDWORK/opt/julia-1.13.1/bin:$PATH

# Create a Julia Kernel for the Odo Tutorial Notebooks
# Use $SCRATCH as it has more storage space than $HOME
export JULIA_DEPOT_PATH=$SCRATCH/.julia-notebooks

# Just install the IJulia package globally
julia -e 'using Pkg; Pkg.add("IJulia")'

# Kernel will show 
julia -e 'using IJulia; installkernel("julia-hpc-sc26", "--project=@.", env=Dict( "PATH" => ENV["PATH"], "LD_LIBRARY_PATH" => ENV["LD_LIBRARY_PATH"], "JULIA_DEPOT_PATH" => ENV["JULIA_DEPOT_PATH"], "JULIA_NUM_THREADS" => "8"))'

# Verify IJulia kernel configuration
cat ~/.local/share/jupyter/kernels/julia-hpc-sc26/kernel.json

echo "Now log into https://jupyter-open.olcf.ornl.gov/ and select the julia-hpc-sc26 1.13 kernel"
echo "Instantiate required packages before each Notebook section"
