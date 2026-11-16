# Exercise 4.2: Energy tuning

## Solution

```julia
results_energy, _ = KernelTuner.tune_kernel(
    "gs_gpu_kernel_tiled!",
    get_cell_by_tag("gs_gpu_kernel_tiled"),
    (n, n),
    arguments,
    tune_params_small,
    restrictions=restrictions,
    grid_div_x=("block_size_x", "Q"), grid_div_y=("block_size_y", "P"),
    lang="Julia",
    observers=[energy_observer],
    objective=energy_key,
    objective_higher_is_better=false,
    verbose=true,
)
```

**Explanation**
1. `observers` takes a vector, as you can use several observers at the same time.
2. `objective` is the name of the measured quantity to optimize. As the name of the energy differs per vendor, we use the variable `energy_key` rather than a fixed string.
3. Lower energy is better, so `objective_higher_is_better=false`. If you set it to `true`, Kernel Tuner reports the *least* efficient configuration as the best one.

## Answers to the questions

1. **Fastest versus most efficient.** This depends on your GPU and on the measurement noise. Often the two are close, or even the same configuration; sometimes the most energy-efficient configuration is slightly slower.
2. **Related, but not the same.** Energy is power times time. A configuration that finishes sooner draws power for a shorter time, so faster configurations tend to use less energy. But the power also differs between configurations: one that keeps fewer cores busy, or moves less data, may run slightly longer at a lower power and use less energy in total.
3. **Noise.** The energy measurements will differ between runs. Energy sensors have a much coarser resolution than timers, the kernel here takes well under a millisecond, and other activity on the GPU influences the measured power. Small differences in energy between configurations are therefore not meaningful: only trust differences that are larger than the variation between runs.
