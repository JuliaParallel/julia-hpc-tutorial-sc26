# Exercise 2.1: Tuning the workgroup size

## Solution

```julia
tune_params = [
    ("block_size_x", [8, 16, 32, 64, 128]),  # workgroup size along x
    ("block_size_y", [1, 2, 4, 8, 16]),      # workgroup size along y
]

results_wg, env_wg = KernelTuner.tune_kernel(
    "gs_gpu_kernel!",
    get_cell_by_tag("gs_gpu_kernel"),
    (n, n),
    arguments,
    tune_params,
    lang="Julia",
    verbose=true,
)

best_wg = (env_wg["best_config"]["block_size_x"], env_wg["best_config"]["block_size_y"])
```

**Explanation**
1. Kernel Tuner recognizes the names `block_size_x` and `block_size_y` as the workgroup dimensions. The order in `tune_params` is also the order of the `Val` arguments in the kernel signature, so `block_size_x` arrives as `BX` and `block_size_y` as `BY`.
2. Each work-item computes one grid point, so we need `n` × `n` work-items: the problem size is `(n, n)`. By default, Kernel Tuner divides it by the workgroup size to get the number of workgroups.
3. `env_wg["best_config"]` holds the parameter values (and the measurements) of the best configuration.

## Answers to the questions

1. **Shape of the fastest workgroup.** Usually the fastest workgroups are wide along `x`, e.g. 32 or more work-items in `x`, rather than square or tall. Julia arrays are column-major, so `U[x, y]` and `U[x+1, y]` are adjacent in memory. When consecutive work-items in a workgroup differ in `x`, they read consecutive addresses, which the GPU combines into a few wide memory transactions (*coalescing*). With a narrow workgroup, e.g. 8 in `x`, the work-items read from several columns at once, which needs more memory transactions. How large the effect is depends on the GPU, so your heatmap may look different: that is exactly why we tune.
2. **The missing `(128, 16)` configuration.** 128 × 16 = 2048 work-items, more than the maximum number of work-items per workgroup on the device (1024 on most GPUs). Kernel Tuner queries this limit from the device and skips configurations that exceed it; with `verbose=true` it reports the skipped configurations.
