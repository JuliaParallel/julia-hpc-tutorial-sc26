using Pkg

Pkg.instantiate()
Pkg.update()
Pkg.build("IJulia")

using IJulia

IJulia.installkernel(
    "julia_gpu_kerneltuning",
    "--project=@.",
    "--threads=auto",
)
notebook()