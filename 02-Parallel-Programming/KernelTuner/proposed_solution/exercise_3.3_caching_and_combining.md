# Exercise 3.3: Caching and combining it all

## Solution

```julia
results_wg_ti, env_wg_ti = with_live_tuning_plot(cachefile) do
    KernelTuner.tune_kernel(
        "gs_gpu_kernel_tiled!",
        get_cell_by_tag("gs_gpu_kernel_tiled"),
        (n, n),
        arguments,
        tune_params,
        restrictions=restrictions,
        grid_div_x=("block_size_x", "Q"), grid_div_y=("block_size_y", "P"),
        lang="Julia",
        verbose=true,
        cache=cachefile,
        strategy=strategy,
        strategy_options=strategy_options,
    )
end
```

**Explanation.** The keyword arguments simply combine the earlier exercises: `restrictions` shrinks the search space before tuning, `strategy` and `strategy_options` decide which configurations to evaluate and when to stop, and `cache` stores every measured configuration in a file.

## Answers to the questions

1. **All at once versus one at a time.** Tuning one group of parameters at a time assumes the groups are independent, while the best tile size can depend on the workgroup size and vice versa. The joint search can therefore find configurations that the step-by-step approach misses. On the other hand, the strategy only evaluates part of the much larger search space within its time limit, so it can also end up slightly worse. Which effect wins depends on your GPU and on the time limit; a longer limit favors the joint search.
2. **Running the cell a second time.** The configurations that are already in the cache file are read from it instead of being benchmarked, which takes almost no time. The strategy starts again, but the configurations it already measured are free, so within the same time limit it gets further and usually evaluates new configurations, which are added to the cache. This is also how you continue a tuning run that was interrupted.
3. **A different strategy.** Most likely you will find a different best configuration, but with a similar time. Strategies are stochastic and explore the search space differently, and the search space usually contains several configurations whose times are within the measurement noise of each other. That's fine: what matters is that the configuration is (nearly) as fast as the optimum.
