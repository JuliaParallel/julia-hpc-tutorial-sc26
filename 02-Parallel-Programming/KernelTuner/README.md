# KernelTuner.jl

## Compatibility
This tutorial has been tested on Ubuntu and macOS systems with AMD, NVIDIA, and Apple GPUs. Intel support coming soon. Windows is currently not supported.

## Quick Start

> [!NOTE]
> If you're running on your own device or the provided AMD HPC Fund cluster, run the below directly (not from a compute node). 
> If you're running on your own cluster, first set the port you want to use in `open_notebook.jl`, then run the below from a compute node. 

`cd` to this directory, then run:
```
julia --project=. open_notebook.jl
```

Make sure to follow the instructions printed to your terminal. 

After you're done, run `jupyter notebook stop`. 
In case you get stuck, see `setup.md`.
