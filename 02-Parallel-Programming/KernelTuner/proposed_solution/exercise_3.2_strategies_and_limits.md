# Exercise 3.2: Optimization strategies and limits

## Solution

```julia
strategy = "genetic_algorithm"     # or e.g. "dual_annealing", "random_sample", "bayes_opt"
```

```julia
strategy_options = [("time_limit", 30)]
```

**Explanation.** `strategy` is the name of one of Kernel Tuner's [optimization strategies](https://kerneltuner.github.io/kernel_tuner/stable/optimization.html#optimization-strategies), as a string. `strategy_options` is a vector of `(name, value)` tuples; `"time_limit"` is the maximum tuning time in seconds. To limit the number of evaluated configurations instead, use `[("max_fevals", 100)]`.

## Answers to the questions

1. **Why there is no guarantee.** Brute force evaluates every configuration, so it cannot miss the best one. A strategy evaluates only part of the search space and decides where to look based on the results so far; the best configuration may be in a part it never visits, for example because it sits in a narrow region surrounded by slow configurations. Most strategies also make random choices, so different runs can end with different results. In practice, strategies usually find a configuration close to the best one in a fraction of the time.
2. **`max_fevals` versus `time_limit`.** A time limit is natural when you have a fixed budget, e.g. "tune for at most ten minutes". But the number of configurations that fit in that time depends on the hardware, the compile time and the system load. `max_fevals` makes runs comparable: for example, to compare strategies with each other fairly, or to get the same tuning effort on different machines.
