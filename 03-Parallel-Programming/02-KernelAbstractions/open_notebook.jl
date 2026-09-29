using Pkg

Pkg.instantiate()

using IJulia

IJulia.installkernel(
    "GPU Julia",
    "--project=@.",
    "--threads=auto",
)
notebook()