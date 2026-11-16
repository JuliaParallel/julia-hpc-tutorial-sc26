# Exercise 2.2: Tuning the tile size

## Solution

```julia
tune_params = [
    ("P", [1, 2, 4, 8, 16]),                    # tile size per work-item along y
    ("Q", [1, 2, 4, 8, 16]),                    # tile size per work-item along x
    ("block_size_x", [best_wg[1]]),             # keep the best workgroup size from Section 2.1 fixed
    ("block_size_y", [best_wg[2]]),
]

results_ti, env_ti = KernelTuner.tune_kernel(
    "gs_gpu_kernel_tiled!",
    get_cell_by_tag("gs_gpu_kernel_tiled"),
    (n, n),
    arguments,
    tune_params,
    grid_div_x=("block_size_x", "Q"), grid_div_y=("block_size_y", "P"),
    lang="Julia",
    verbose=true
)
```

**Explanation**
1. Every tunable parameter takes a list of values, so a fixed value is written as a list with one element: `[best_wg[1]]`, not `best_wg[1]`.
2. Each workgroup now covers `block_size_x * Q` points along `x` and `block_size_y * P` points along `y`. Kernel Tuner computes the number of workgroups in each dimension by dividing the problem size by the product of the parameters listed in `grid_div_x` and `grid_div_y`, rounded up: `cld(n, block_size_x * Q)` workgroups along `x`.

## Answers to the questions

1. **The fastest tile size** depends on your GPU. Moderate tiles, such as 2 or 4 points in each direction, often beat the kernel without tiling (`P = Q = 1`), because each work-item loads $(Q+2)(P+2)$ values for $QP$ points instead of 5 values per point. If tiling does not help on your GPU, its caches already serve most of the repeated neighbor reads.
2. **Large tiles are slow** because they need too many registers. With `P = Q = 16`, each work-item keeps two halo tiles of 18 × 18 values and two result tiles of 16 × 16 values: 1160 floats, far more than the at most 255 registers a work-item can use on NVIDIA GPUs (and similar limits elsewhere). The values that do not fit *spill* to much slower memory, and as each work-item uses more registers, fewer work-items can run at the same time to hide memory latency. There are also fewer work-items in total: with 16 × 16 tiles, only 64 × 64 = 4096 work-items cover the 1024 × 1024 grid, too few to keep a GPU busy.
3. **Without `grid_div_x` and `grid_div_y`**, Kernel Tuner divides the problem size only by the workgroup size, and launches `n × n` work-items, `P × Q` times more than needed. The extra work-items start beyond the grid. The kernel's bounds checks keep them from writing anything, so the result is still correct, but they waste time, and Kernel Tuner would report misleadingly slow times for larger tiles. The opposite mistake, dividing by too much, launches too few work-items and leaves part of the grid uncomputed. That gives a wrong result, which output verification (Section 4.1) catches.
