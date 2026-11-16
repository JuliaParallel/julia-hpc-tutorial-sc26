# KernelAbstractions.jl

## Overview

[KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl) lets you write a GPU kernel once, in plain Julia, and run it on many kinds of hardware: NVIDIA (CUDA), AMD (AMDGPU), Apple (Metal), Intel (oneAPI) GPUs, and multithreaded CPUs. You mark a function with `@kernel`, use `@index` to find which element each thread works on, and pick the backend when you launch it. The same code then compiles to native code for the device you chose.

This part of the tutorial teaches GPU programming through that portable interface, across three notebooks:

1. **`gpu_introduction.ipynb`: first steps on the GPU.** Choosing a backend, GPU arrays and array operations, why looping over device arrays from the host is slow, writing your first kernel (vector addition, then SAXPY as an exercise), the execution hierarchy and global indexing, timing kernels, and race conditions in global memory.
2. **`matrix_multiplication.ipynb`: memory access and performance.** Matrix transpose and matrix multiplication on the CPU, with the vendor library, and as naive kernels, then faster tiled kernels that use shared (local) memory. Includes exercises building up to a tiled matrix multiplication.
3. **`gray-scott.ipynb`: a real application.** The Gray-Scott reaction-diffusion model, solved with a 2D 5-point stencil: a CPU reference version first, then a KernelAbstractions version, timed on the same grids on CPU and GPU.

Each notebook asks you to choose a backend (`DEV`) in its second cell, so you can follow along on a laptop or on an AMD GPU node of the cluster. Completed notebooks are in `solved_notebooks/`, and a solution for each exercise is in `proposed-solution/`.

## Requirements

```
julia --project=. open_notebook.jl
```

## Environment setup for GPU computing

There are two options: your own machine (macOS or NVIDIA GPU), or the AMD HPC Fund cluster.

### Quick start: one command

`hpcfund.sh` (in this folder) runs all the steps below for you. On macOS, Linux or WSL:

```bash
curl -fsSLO https://raw.githubusercontent.com/JuliaParallel/julia-hpc-tutorial-sc26/main/02-Parallel-Programming/KernelAbstractions/hpcfund.sh
chmod +x hpcfund.sh

./hpcfund.sh local                # your own Mac / GPU machine
./hpcfund.sh cluster <username>   # AMD HPC Fund cluster
```

In cluster mode the script logs in once, sets up Julia and the **GPU Julia** kernel, submits the notebook job, tunnels Jupyter to your laptop and opens your browser. Press **Ctrl-C** when you are done: it cancels the job for you. Add a folder to use other notebooks, e.g. `./hpcfund.sh cluster <username> KernelTuner/proposed_solution`. Run `./hpcfund.sh --help` for options.

The manual steps follow.

### Local machine: if you have a MacBook or NVIDIA GPU

1. Install [Julia](https://julialang.org/downloads/).
2. Clone the repo and enter the KernelAbstractions project:

   ```bash
   git clone https://github.com/JuliaParallel/julia-hpc-tutorial-sc26.git
   cd julia-hpc-tutorial-sc26/02-Parallel-Programming/KernelAbstractions
   ```

3. Open Jupyter from that folder (installs the **GPU Julia** IJulia kernel and starts the notebook server):

   ```bash
   julia --project=. open_notebook.jl
   ```

4. In Jupyter, pick kernel **GPU Julia** and open a notebook. In each notebook, set `DEV` in the **second cell** (`:CUDA` (NVIDIA), `:AMDGPU`, `:Metal`, or `:CPU`) before running `Pkg.add`, to match your hardware.

### AMD HPC cluster: if you do not have a MacBook or NVIDIA GPU

1. Open a terminal and connect to the cluster:

   ```bash
   ssh username@hpcfund.amd.com
   ```

2. Install Julia, clone the git repo, and navigate to the correct folder:

   ```bash
   curl -fsSL https://install.julialang.org | sh
   source ~/.bashrc
   git clone https://github.com/JuliaParallel/julia-hpc-tutorial-sc26.git
   cd julia-hpc-tutorial-sc26/02-Parallel-Programming/KernelAbstractions
   ```

3. Start Jupyter on a compute node, either automated (A) or manually (B).

   **A. Automated.** From the folder of step 2, run the line below. It installs the **GPU Julia** kernel, submits the notebook job, and prints the SSH tunnel command and URL to use from your laptop:

   ```bash
   julia --project=. open_notebook.jl
   ```

   **B. Manual.** Request a node with a Jupyter notebook attached (replace `<job-id>`) and look at its output. If the out file does not exist, your job may be queued; check with `squeue --me`. Repeat these steps for every new folder you work in.

   ```bash
   cp ../../amdhpcfund.job.notebook job.notebook   # or: cp /work1/proj_sc26julia/shared/job.notebook .
   julia -e 'using Pkg; Pkg.add("IJulia")'
   julia --project=. -e 'using Pkg; Pkg.instantiate(); using IJulia; IJulia.installkernel("GPU Julia", "--project=@.", "--threads=auto")'
   sbatch job.notebook
   cat job.<job-id>.out
   ```

4. Follow the instructions in the job output to open an SSH tunnel from your laptop, then open the Jupyter URL in your browser.

5. In Jupyter, pick kernel **GPU Julia** and open a notebook. In each notebook, set `DEV` in the **second cell** (`:CUDA` (NVIDIA), `:AMDGPU`, `:Metal`, or `:CPU`) before running `Pkg.add`, to match your hardware.

6. When you are done, do not forget to close your allocation!

   ```bash
   scancel --me
   ```

## Credits

The material in this folder is based on tutorials by Valentin Churavy, Rabab Alomairy, and Collin Wittenstein.
