# Exercise 4.4: User-defined metrics

## Solution

```julia
metrics = [
    "GFLOP/s" => "$(flops_per_point * grid_points / 1e9) / (time / 1e3)",   # time is in milliseconds
    "GB/s" => "$(bytes_per_point * grid_points / 1e9) / (time / 1e3)",
    "work-items" => p -> p["block_size_x"] * p["block_size_y"],
]
```

**Explanation.** The bandwidth is the amount of data moved in GB divided by the time in seconds. `bytes_per_point * grid_points / 1e9` is computed once, in Julia, and inserted into the string as a number; `time` is filled in by Kernel Tuner for every configuration, in milliseconds, so we divide it by `1e3`. Alternatively, as a Julia function: `"GB/s" => p -> bytes_per_point * grid_points / 1e9 / (p["time"] / 1e3)`.

## Answers to the questions

1. **Compared to the peak bandwidth.** For reference, an Apple M2 Pro has a peak of 200 GB/s, an NVIDIA A100 about 1.6–2.0 TB/s and an AMD MI250X 1.6 TB/s per GCD. The kernel will likely stay well below the peak. One reason is that our metric counts the *minimum* data that must be moved, while the actual traffic is higher. Another is that a 1024 × 1024 grid takes only a fraction of a millisecond, which is not enough to fully occupy a large GPU.
2. **Compute- or memory-bound?** The arithmetic intensity is 34 FLOP / 16 bytes ≈ 2 FLOP per byte. A GPU can do roughly 10 to 50 floating-point operations in the time it moves one byte from memory (its peak FLOP/s divided by its peak bandwidth). The kernel is far below that, so it is limited by memory bandwidth. This is why the GB/s metric is the more relevant one here, and why reducing the data moved, such as with 16-bit storage in Exercise 4.3, can help.
3. **Time per step in the simulation.** On top of the kernel time, each time step pays for launching the kernel from the host. For a kernel this short, that overhead can be as large as the kernel itself. A larger grid, or computing several time steps per kernel launch, then helps more than further tuning.

## Bonus: GFLOP/J

The energy observer adds the measured energy, in joules, to the results under the name `energy_key`, so a metric can divide the work by it:

```julia
metrics_energy = ["GFLOP/J" => "$(flops_per_point * grid_points / 1e9) / $energy_key"]

results_gflopj, env_gflopj = KernelTuner.tune_kernel(
    "gs_gpu_kernel_tiled!",
    get_cell_by_tag("gs_gpu_kernel_tiled"),
    (n, n),
    arguments,
    tune_params_small,
    restrictions=restrictions,
    grid_div_x=("block_size_x", "Q"), grid_div_y=("block_size_y", "P"),
    lang="Julia",
    observers=[energy_observer],
    metrics=metrics_energy,
    objective="GFLOP/J",
    objective_higher_is_better=true,    # more work per joule is better
    verbose=true,
)
env_gflopj["best_config"]
```

Note that `$energy_key` inserts the *name* of the energy field, e.g. `nvml_energy`, into the expression. Optimizing for GFLOP/J gives the same best configuration as minimizing the energy, as the work per step is the same for all configurations. The metric makes the result easier to compare across kernels and GPUs.
