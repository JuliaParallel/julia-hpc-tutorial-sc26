# Exercise 5: All combined

## Solution

```julia
results_combined, env_combined = with_live_tuning_plot(cachefile_energy; plotter=plot_time_energy, energy_key=energy_key) do
    KernelTuner.tune_kernel(
        "gs_gpu_kernel_tiled!",
        get_cell_by_tag("gs_gpu_kernel_tiled"),
        (n, n),
        arguments,
        tune_params,                    # the large search space of Section 3.3
        restrictions=restrictions,
        grid_div_x=("block_size_x", "Q"), grid_div_y=("block_size_y", "P"),
        verbose=true,
        observers=[energy_observer],        # measure energy (Section 4.2)
        objective=energy_key,               # and optimize for it
        objective_higher_is_better=false,   # lower energy is better
        answer=answer,                      # verify the output (Section 4.1)
        atol=1e-5,
        strategy=strategy,                  # the strategy and limit (Section 3.2)
        strategy_options=strategy_options,
        cache=cachefile_energy,             # cache the results (Section 3.3)
    )
end
```

**Explanation.** Every feature is a keyword argument, so they combine freely. Two details matter here. Measuring energy takes about a second per configuration, so without a strategy and limit, Kernel Tuner would brute-force all 525 configurations and take around ten minutes. And we use a separate cache file, because the results cached in Section 3.3 contain no energy measurements.

## Answers to the questions

1. **The Pareto front.** The numbers depend on your GPU and on the measurement noise; the cell below the tuning prints the trade-offs for you. Typically, the front holds only a few configurations, and the most energy-efficient one trades a small percentage of time for a small percentage of energy. If the front holds a single configuration, the fastest configuration is also the most efficient one.
2. **Choosing a configuration.** When time to solution is what limits you, e.g. a deadline or an allocation measured in node hours, choose the fastest configuration, at the start of the front. Under an energy or power budget, choose one further along the front, accepting a longer run time for the energy it saves. Every configuration on the front is a reasonable choice; which one is best depends on how you weigh time against energy.
3. **Configurations above the front** are *dominated*: some configuration on the front is both faster and uses less energy, so there is no reason to prefer them, whatever your weighing of time and energy. Keep the noise in mind, though: configurations just above the front may be just as good within the measurement precision.
