# Julia on HPC Frontier

This material is based on the Julia for HPC training series offered at Oak Ridge National Laboratory (ORNL) Leadership Computing Facility (OLCF). Tutorial [website](https://juliaornl.github.io/TutorialJuliaHPC/).

In this module, we will learn:

1. Running a Julia application [Gray-Scott](https://github.com/JuliaORNL/GrayScott.jl) on the Frontier training system, [Odo](https://docs.olcf.ornl.gov/systems/odo_user_guide.html) 32 nodes, each node equipped with AMD's MI250X.

2. Deploying Julia applications on the Frontier system, Julia packaging and HPC module system interactions
   
4. Composing Julia applications for HPC environments: structuring code, managing dependencies, 

5. HPC components in Julia: 
   
   - Communication: Message passing interface [MPI.jl](https://github.com/JuliaParallel/MPI.jl)
   - Parallel I/O: [ADIOS2.jl](https://github.com/eschnett/ADIOS2.jl)
   - CPU/GPU vendor-neutral computing with: [JACC.jl](https://github.com/JuliaGPU/JACC.jl) and [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl) and vendor backends: AMDGPU.jl, Threads and other JuliaGPU backends: CUDA.jl, oneAPI.jl, Metal.jl
   - Profiling Julia code with [rocprof](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3.html)
  

Participants with an Odo account can start following the tutorial [steps](https://juliaornl.github.io/TutorialJuliaHPC/applications/GrayScott/04-OdoFrontier.html)