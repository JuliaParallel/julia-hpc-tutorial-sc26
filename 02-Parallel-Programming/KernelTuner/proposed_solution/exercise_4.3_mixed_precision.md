# Exercise 4.3: Accuracy tuning and mixed precision

## Solution

```julia
storage(A) = KernelTuner.Tunable("storage_bits", Dict(32 => Float32.(A), 16 => Float16.(A)))
```

```julia
accuracy_observer = AccuracyObserver("RMSE")
```

```julia
fastest_accurate = argmin(r -> r["time"], filter(r -> r["error"] <= max_error, results_mixed))
```

**Explanation**
1. `KernelTuner.Tunable(param, Dict(value => array))` lets Kernel Tuner pick the argument based on the value of a tunable parameter. For every configuration, it passes the `Float32` copy when `storage_bits` is 32 and the `Float16` copy when it is 16. The kernel is generic, so the same code works with both array types.
2. `AccuracyObserver("RMSE")` compares the outputs to `answer` and adds the root-mean-square error to the results as `"error"`. Instead of `"RMSE"`, you can pass `"MAE"`, `"relative error"` or your own function `(answer, output) -> error`, e.g. `AccuracyObserver((answer, output) -> maximum(abs.(answer .- output)))` for the maximum error.
3. First keep only the configurations whose error is within the tolerance, then take the one with the lowest time, just like `fastest_overall` but on the filtered results.

## Answers to the questions

1. **Within the tolerance.** Only the configurations with 32-bit storage *and* 32-bit compute. `Float32` has a relative precision of about $10^{-7}$, `Float16` of about $10^{-3}$: as soon as either the storage or the arithmetic uses 16 bits, the error grows by several orders of magnitude, well above $10^{-6}$.
2. **Are 16-bit configurations faster?** This depends on the GPU. 16-bit storage halves the data that is moved, which helps a memory-bound kernel such as this one (see Exercise 4.4). 16-bit arithmetic is faster on GPUs with fast `Float16` units, but on others it is no faster, or even slower because of the conversions between precisions. And when the kernel is small, the launch overhead and occupancy matter more than the precision.
3. **A full simulation.** Errors accumulate over thousands of time steps, and the Gray-Scott model is sensitive to small perturbations: patterns that form can come out completely different. 16-bit storage has an additional problem. The change per time step, `DU * dt`, is often much smaller than the spacing between `Float16` values around `u` (about $5 \cdot 10^{-4}$ for values near 1), so the update is rounded away and the simulation stalls. Accuracy should therefore be judged on the application's result, not only on a single step.
