# Answers to the questions in the exercises

Answers to the questions in `kerneltuner.ipynb`, collected in one place. The per-exercise files in this folder contain the same answers, together with the code solutions.

## Exercise 2.1: Tuning the workgroup size

The heatmap shows the time of each configuration (lower is better).

*1. Is the fastest workgroup shape square, or wide in one direction? Can you explain why? Hint: Julia arrays are column-major, so `U[x, y]` and `U[x+1, y]` are next to each other in memory.*

**Shape of the fastest workgroup.** Usually the fastest workgroups are wide along `x`, e.g. 32 or more work-items in `x`, rather than square or tall. Julia arrays are column-major, so `U[x, y]` and `U[x+1, y]` are adjacent in memory. When consecutive work-items in a workgroup differ in `x`, they read consecutive addresses, which the GPU combines into a few wide memory transactions (*coalescing*). With a narrow workgroup, e.g. 8 in `x`, the work-items read from several columns at once, which needs more memory transactions. How large the effect is depends on the GPU, so your heatmap may look different: that is exactly why we tune.

*2. The configuration `(128, 16)` is missing from the heatmap. Why? Hint: look at the output of `tune_kernel`.*

**The missing `(128, 16)` configuration.** 128 × 16 = 2048 work-items, more than the maximum number of work-items per workgroup on the device (1024 on most GPUs). Kernel Tuner queries this limit from the device and skips configurations that exceed it; with `verbose=true` it reports the skipped configurations.

## Exercise 2.2: Tuning the tile size

*1. Which tile size is the fastest? Is it faster than the best configuration without tiling from Exercise 2.1 (compare the best times printed by Kernel Tuner)?*

**The fastest tile size** depends on your GPU. Moderate tiles, such as 2 or 4 points in each direction, often beat the kernel without tiling (`P = Q = 1`), because each work-item loads $(Q+2)(P+2)$ values for $QP$ points instead of 5 values per point. If tiling does not help on your GPU, its caches already serve most of the repeated neighbor reads.

*2. Why are the largest tiles, such as `P = Q = 16`, slow?*

**Large tiles are slow** because they need too many registers. With `P = Q = 16`, each work-item keeps two halo tiles of 18 × 18 values and two result tiles of 16 × 16 values: 1160 floats, far more than the at most 255 registers a work-item can use on NVIDIA GPUs (and similar limits elsewhere). The values that do not fit *spill* to much slower memory, and as each work-item uses more registers, fewer work-items can run at the same time to hide memory latency. There are also fewer work-items in total: with 16 × 16 tiles, only 64 × 64 = 4096 work-items cover the 1024 × 1024 grid, too few to keep a GPU busy.

*3. What would happen if you left out `grid_div_x` and `grid_div_y`? Would the result still be correct?*

**Without `grid_div_x` and `grid_div_y`**, Kernel Tuner divides the problem size only by the workgroup size, and launches `n × n` work-items, `P × Q` times more than needed. The extra work-items start beyond the grid. The kernel's bounds checks keep them from writing anything, so the result is still correct, but they waste time, and Kernel Tuner would report misleadingly slow times for larger tiles. The opposite mistake, dividing by too much, launches too few work-items and leaves part of the grid uncomputed. That gives a wrong result, which output verification (Section 4.1) catches.

## Exercise 3.1: Restrictions

1. The upper bound of 1024 work-items is the hardware maximum on most GPUs. Why is a lower bound of 32 work-items sensible?
2. How many of the workgroup sizes from Exercise 2.1 satisfy this restriction?

*1. The upper bound of 1024 work-items is the hardware maximum on most GPUs. Why is a lower bound of 32 work-items sensible?*

**The lower bound of 32.** GPUs execute the work-items of a workgroup in groups of a fixed size: 32 on NVIDIA (warps) and Apple (SIMD-groups), 64 on AMD data center GPUs (wavefronts). A workgroup with fewer work-items than that still occupies a whole group, leaving the rest of its lanes idle. Such configurations are practically always slow, so there is no point in trying them.

*2. How many of the workgroup sizes from Exercise 2.1 satisfy this restriction?*

**Remaining workgroup sizes.** Of the 25 workgroup sizes of Exercise 2.1, 21 remain. Removed are 8 × 1, 8 × 2 and 16 × 1 (fewer than 32 work-items) and 128 × 16 (more than 1024). With the 25 tile sizes of Section 3.3, this leaves 21 × 25 = 525 of the 625 configurations, which the cell at the start of Section 3.3 confirms.

## Exercise 3.2: Optimization strategies and limits

1. Why is an optimization strategy not guaranteed to find the best configuration, while brute force is?
2. When would you rather limit the number of evaluated configurations with `max_fevals` than the tuning time with `time_limit`?

*1. Why is an optimization strategy not guaranteed to find the best configuration, while brute force is?*

**Why there is no guarantee.** Brute force evaluates every configuration, so it cannot miss the best one. A strategy evaluates only part of the search space and decides where to look based on the results so far; the best configuration may be in a part it never visits, for example because it sits in a narrow region surrounded by slow configurations. Most strategies also make random choices, so different runs can end with different results. In practice, strategies usually find a configuration close to the best one in a fraction of the time.

*2. When would you rather limit the number of evaluated configurations with `max_fevals` than the tuning time with `time_limit`?*

**`max_fevals` versus `time_limit`.** A time limit is natural when you have a fixed budget, e.g. "tune for at most ten minutes". But the number of configurations that fit in that time depends on the hardware, the compile time and the system load. `max_fevals` makes runs comparable: for example, to compare strategies with each other fairly, or to get the same tuning effort on different machines.

## Exercise 3.3: Caching and combining it all

*1. How does tuning everything at once compare to tuning one group of parameters at a time, as in Section 2? Run the cell below to compare. Keep in mind that the strategy has only evaluated part of the search space within its time limit.*

**All at once versus one at a time.** Tuning one group of parameters at a time assumes the groups are independent, while the best tile size can depend on the workgroup size and vice versa. The joint search can therefore find configurations that the step-by-step approach misses. On the other hand, the strategy only evaluates part of the much larger search space within its time limit, so it can also end up slightly worse. Which effect wins depends on your GPU and on the time limit; a longer limit favors the joint search.

*2. Run the tuning cell above a second time. What happens, and why?*

**Running the cell a second time.** The configurations that are already in the cache file are read from it instead of being benchmarked, which takes almost no time. The strategy starts again, but the configurations it already measured are free, so within the same time limit it gets further and usually evaluates new configurations, which are added to the cache. This is also how you continue a tuning run that was interrupted.

*3. Change the strategy in Exercise 3.2 and run the tuning again (remove the cache file first). Do you find the same best configuration?*

**A different strategy.** Most likely you will find a different best configuration, but with a similar time. Strategies are stochastic and explore the search space differently, and the search space usually contains several configurations whose times are within the measurement noise of each other. That's fine: what matters is that the configuration is (nearly) as fast as the optimum.

## Exercise 4.1: Output verification

*1. Change `atol` to `0` and run the cell again. What happens, and why?*

**`atol=0`.** Kernel Tuner will most likely stop with a verification error, even though the kernel is correct. The GPU and the CPU do not round exactly the same: the compilers may evaluate the floating-point operations in a different order or fuse a multiply and an add into one instruction, which rounds once instead of twice. The results then differ in the last bits. That's why we allow a small tolerance; it should be well above these rounding differences but far below the size of an actual bug.

*2. Why do we use the simple kernel `gs_gpu_kernel!` to compute the reference, and not the tiled kernel that we are tuning?*

**Why the simple kernel as reference.** The reference must be independent of what we test. If we computed it with the tiled kernel, we would only check that all configurations agree with *one* configuration of the same code, and a bug in the tiling logic would go unnoticed. The simple kernel is easier to get right and was already validated against the CPU implementation in the KernelAbstractions tutorial. Running it on the CPU also avoids any GPU-specific problem in the reference.

## Exercise 4.2: Energy tuning

*1. Is the fastest configuration also the most energy-efficient one on your GPU?*

**Fastest versus most efficient.** This depends on your GPU and on the measurement noise. Often the two are close, or even the same configuration; sometimes the most energy-efficient configuration is slightly slower.

*2. Time and energy are related: why? And why are they not the same?*

**Related, but not the same.** Energy is power times time. A configuration that finishes sooner draws power for a shorter time, so faster configurations tend to use less energy. But the power also differs between configurations: one that keeps fewer cores busy, or moves less data, may run slightly longer at a lower power and use less energy in total.

*3. Run the energy tuning again. Do you get the same energy measurements? What does that mean for how you interpret small differences?*

**Noise.** The energy measurements will differ between runs. Energy sensors have a much coarser resolution than timers, the kernel here takes well under a millisecond, and other activity on the GPU influences the measured power. Small differences in energy between configurations are therefore not meaningful: only trust differences that are larger than the variation between runs.

## Exercise 4.3: Accuracy tuning and mixed precision

*1. Which combinations of storage and compute precision stay within the tolerance of $10^{-6}$?*

**Within the tolerance.** Only the configurations with 32-bit storage *and* 32-bit compute. `Float32` has a relative precision of about $10^{-7}$, `Float16` of about $10^{-3}$: as soon as either the storage or the arithmetic uses 16 bits, the error grows by several orders of magnitude, well above $10^{-6}$.

*2. Are the 16-bit configurations faster than the 32-bit ones on your GPU? Why could this differ between GPUs?*

**Are 16-bit configurations faster?** This depends on the GPU. 16-bit storage halves the data that is moved, which helps a memory-bound kernel such as this one (see Exercise 4.4). 16-bit arithmetic is faster on GPUs with fast `Float16` units, but on others it is no faster, or even slower because of the conversions between precisions. And when the kernel is small, the launch overhead and occupancy matter more than the precision.

*3. This is the error of a single time step. Why would the error of a full simulation of thousands of steps be a bigger concern, especially with 16-bit storage?*

**A full simulation.** Errors accumulate over thousands of time steps, and the Gray-Scott model is sensitive to small perturbations: patterns that form can come out completely different. 16-bit storage has an additional problem. The change per time step, `DU * dt`, is often much smaller than the spacing between `Float16` values around `u` (about $5 \cdot 10^{-4}$ for values near 1), so the update is rounded away and the simulation stalls. Accuracy should therefore be judged on the application's result, not only on a single step.

## Exercise 4.4: User-defined metrics

*1. Look up the peak memory bandwidth of your GPU in its specifications. How close does the best configuration get?*

**Compared to the peak bandwidth.** For reference, an Apple M2 Pro has a peak of 200 GB/s, an NVIDIA A100 about 1.6–2.0 TB/s and an AMD MI250X 1.6 TB/s per GCD. The kernel will likely stay well below the peak. One reason is that our metric counts the *minimum* data that must be moved, while the actual traffic is higher. Another is that a 1024 × 1024 grid takes only a fraction of a millisecond, which is not enough to fully occupy a large GPU.

*2. The kernel performs 34 floating-point operations for every 16 bytes it moves. Is it limited by computation or by memory bandwidth?*

**Compute- or memory-bound?** The arithmetic intensity is 34 FLOP / 16 bytes ≈ 2 FLOP per byte. A GPU can do roughly 10 to 50 floating-point operations in the time it moves one byte from memory (its peak FLOP/s divided by its peak bandwidth). The kernel is far below that, so it is limited by memory bandwidth. This is why the GB/s metric is the more relevant one here, and why reducing the data moved, such as with 16-bit storage in Exercise 4.3, can help.

*3. Kernel Tuner only measures the time the kernel runs on the GPU. What else determines the time per step in the actual simulation?*

**Time per step in the simulation.** On top of the kernel time, each time step pays for launching the kernel from the host. For a kernel this short, that overhead can be as large as the kernel itself. A larger grid, or computing several time steps per kernel launch, then helps more than further tuning.

## Exercise 5: All combined

*1. How many configurations are on the Pareto front? How much time do you lose by choosing the most energy-efficient configuration instead of the fastest, and how much energy do you save?*

**The Pareto front.** The numbers depend on your GPU and on the measurement noise; the cell below the tuning prints the trade-offs for you. Typically, the front holds only a few configurations, and the most energy-efficient one trades a small percentage of time for a small percentage of energy. If the front holds a single configuration, the fastest configuration is also the most efficient one.

*2. Which configuration would you choose if you are limited by time to solution? And if you run on a system with a strict energy or power budget?*

**Choosing a configuration.** When time to solution is what limits you, e.g. a deadline or an allocation measured in node hours, choose the fastest configuration, at the start of the front. Under an energy or power budget, choose one further along the front, accepting a longer run time for the energy it saves. Every configuration on the front is a reasonable choice; which one is best depends on how you weigh time against energy.

*3. Why can configurations far above the front be discarded?*

**Configurations above the front** are *dominated*: some configuration on the front is both faster and uses less energy, so there is no reason to prefer them, whatever your weighing of time and energy. Keep the noise in mind, though: configurations just above the front may be just as good within the measurement precision.
