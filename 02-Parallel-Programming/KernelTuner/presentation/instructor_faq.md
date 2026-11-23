# Kernel Tuner tutorial: likely questions and answers

Notes for the presenter. Section numbers refer to `kerneltuner.ipynb`.

---

## Performance results

### Why is the best GB/s only 1/3 to 1/4 of the peak bandwidth of the GPU?

Several effects stack up. Most of them have nothing to do with tuning:

1. **The spec-sheet peak is not attainable.** Even a pure streaming kernel (STREAM / BabelStream) typically reaches about 75–90% of the theoretical bandwidth. The realistic ceiling is therefore already lower than the number on the box.
2. **The kernel is short.** One step moves at least 1022² × 16 B ≈ 16.7 MB. At peak bandwidth that would take:

   | GPU | Peak bandwidth | Ideal time per step | At 1/3–1/4 of peak |
   |---|---|---|---|
   | NVIDIA RTX A4000 | 448 GB/s | 37 µs | 110–150 µs |
   | AMD Instinct MI210 | 1.6 TB/s | 10 µs | 30–40 µs |
   | Apple M5 Pro | *fill in from Apple's specs* | | |

   At these time scales, fixed costs weigh heavily: starting up the workgroups, the "tail" where the last workgroups run on a half-empty GPU, and the clocks not ramping up fully for short bursts. This hits the MI210 hardest, because its ideal time is only about 10 µs.
3. **Not enough memory requests in flight.** To saturate memory, a GPU needs about *bandwidth × latency* bytes in flight (Little's law): on the order of a few hundred kB to 1 MB. Each work-item issues a handful of scalar 4-byte loads. Index arithmetic, bounds checks and, for the tiled kernel, lower occupancy due to register use all limit how many requests are outstanding.
4. **Extra overhead from the index computation.** `@index(Global, NTuple)` and KernelAbstractions' implicit range check add integer arithmetic to every work-item, including divisions to turn a linear index into `(x, y)`. In a kernel with this little work per byte, that overhead is noticeable.
5. **The metric undercounts the traffic.** 16 B per point assumes every neighbor read hits cache. Halo reads and, in the tiled kernel, the overlapping tile borders ($(Q+2)(P+2)$ loads per $QP$ points) add traffic that the metric does not count.
6. **Strided access in the tiled kernel.** With `Q > 1`, neighboring work-items access addresses `Q` elements apart, so loads coalesce worse. This is also why `Q = 1` often wins.

**How to show it live, or check it afterwards:**
- Increase `n` to 4096 or 8192. If the GB/s fraction rises substantially, the short kernel and its fixed costs were the main cause.
- Tune a plain copy kernel (`B[i] = A[i]`) with Kernel Tuner to get a realistic ceiling for your GPU.
- Measure the actual DRAM traffic with a profiler: Nsight Compute (NVIDIA), `rocprof` / Omniperf (AMD), or the Metal System Trace in Xcode Instruments (Apple).

### Why is the best GFLOP/s only 1/20 to 1/40 of the peak FLOP/s?

Because the kernel is **memory-bound**: the low GFLOP/s is expected, not a sign of bad tuning.

- Both metrics divide a fixed amount of work by the same measured time. That makes **GFLOP/s = (34 / 16) × GB/s ≈ 2.1 × GB/s for every configuration**, so the GFLOP/s number carries no information beyond GB/s.
- The kernel's *arithmetic intensity* is 34 FLOP / 16 B ≈ 2.1 FLOP/B. A GPU's *ridge point* in the roofline model (peak FLOP/s ÷ peak bandwidth) is much higher:

  | GPU | Peak FP32 | Peak bandwidth | Ridge point | Max. fraction of peak FLOP/s *at full bandwidth* |
  |---|---|---|---|---|
  | RTX A4000 | 19.2 TFLOP/s | 448 GB/s | ≈ 43 FLOP/B | 2.1 / 43 ≈ 1/20 |
  | MI210 | 22.6 TFLOP/s (vector) | 1.6 TB/s | ≈ 14 FLOP/B | 2.1 / 14 ≈ 1/7 |
  | M5 Pro | *fill in* | *fill in* | | |

  So even at 100% of the bandwidth, the A4000 cannot exceed about 1/20 of its FLOP peak on this kernel. Multiply that by the 1/3–1/4 bandwidth fraction from the previous question to get the expected FLOP fraction. If your measured numbers don't fit this, check which peak you compared against. For example, the MI210 also lists 45.3 TFLOP/s for packed FP32, and the A4000's 19.2 TFLOP/s assumes Ampere's dual-issue FP32 throughout.
- The peak FLOP/s also assumes a fused multiply-add on every lane in every cycle. This kernel has divisions (`/ dx`), which take several instructions but count as one FLOP, plus plenty of non-FLOP integer and memory instructions.

**Take-away for the audience:** for a stencil like this, optimize and report bandwidth, not FLOP/s. Higher FLOP/s would only be reachable by doing more work per byte, e.g. several time steps per kernel launch (temporal blocking).

### Why does tuning only give a small speedup over the default (16 × 16) here?

- The kernel is simple and memory-bound, and a 16 × 16 workgroup is a reasonable default on most GPUs. There isn't much room left.
- Kernel Tuner reports kernel time only. In the full simulation, the launch overhead per step comes on top and dilutes the gain.
- The value of tuning shows more in (a) the spread between the worst and best configurations in the heatmaps, which tells you what a bad default can cost, (b) kernels with many interacting parameters, and (c) portability: the best configuration differs per GPU, and tuning finds it without manual work.

### Why is the best configuration different on each GPU?

The hardware differs: SIMD width (32 on NVIDIA and Apple; 64 on AMD CDNA such as the MI210), register file and cache sizes, memory subsystem, maximum workgroup size, and how the compiler handles unrolling and private arrays. That is the main argument for auto-tuning rather than hard-coding one configuration.

### Is the best configuration for `n = 1024` also the best for other grid sizes?

Not necessarily. The problem size changes the number of workgroups relative to the number of compute units, cache behavior and tail effects. Tune for the size you will actually run. Note that the cache file does not record the problem size (see "Caching" below).

### Why is tiling often not faster, or best with `Q = 1`?

- The caches already serve most repeated neighbor reads, so tiling saves less than the "5 loads → $(Q+2)(P+2)/(QP)$" count suggests.
- `Q > 1` makes neighboring work-items access memory with a stride of `Q`, which hurts coalescing. Tiling along `y` (`P > 1`) does not have this problem, as `y` is the slow dimension.
- Large tiles need many registers: they spill, and lower occupancy.
- Fewer work-items in total: with large tiles, there may be too few to fill the GPU.

### On the MI210, should the lower bound in the restriction be 64?

Yes, CDNA GPUs execute 64 work-items per wavefront, so workgroups of 32 leave half of each wavefront idle. Restrictions can reflect the target hardware; 32 is a portable lower bound.

### Why did 16-bit storage/compute not speed things up on my GPU?

16-bit storage halves the bytes moved, but the gain is limited by the effects in the first question: overhead and too few requests in flight, rather than the number of bytes. 16-bit compute only helps on GPUs with fast FP16 units, and the conversions between `Float16` and `Float32` add instructions. And as Section 4.3 shows, 16-bit storage is too inaccurate for this simulation anyway.

---

## Measurements

### How reliable are the timings? Why does the best configuration change between runs?

Kernel Tuner runs every configuration 7 times by default (`iterations=...`) and reports the mean; the individual runs are in `times`. Configurations within a few percent of each other are effectively tied, so the "best" one can change between runs. For more stable results, increase `iterations`, make sure nothing else runs on the GPU, and compare the spread in `times`.

### Why are the energy measurements so noisy?

Energy sensors (NVML, AMD SMI, `powermetrics`) have a much coarser resolution in time than kernel timers, while these kernels take microseconds. The observers therefore measure each configuration for about a second, and the result still includes idle power and other activity on the GPU. Only differences larger than the run-to-run variation are meaningful.

### Why does `powermetrics` need `sudo` on macOS?

It reads hardware counters that macOS only exposes to administrators. The notebook explains how to add a `sudoers` entry for passwordless access.

### Why is tuning so slow when the kernels take microseconds?

Each configuration is compiled separately (`Val` parameters make it a new method specialization), and Julia's GPU compilation takes far longer than the kernel runs. On top of that come the transfers between Python (Kernel Tuner) and Julia, the 7 benchmark iterations, and, with energy observers, about a second per configuration. This is why strategies, limits and caching matter.

### Why does verification with `atol=0` fail although the kernel is correct?

The CPU and GPU round differently: different evaluation order, fused multiply-adds. The tolerance absorbs these last-bit differences; it should be far below the size of a real bug.

---

## Using Kernel Tuner

### Why do the tunable parameters have to be `Val` arguments?

They must be compile-time constants: `@private` arrays need a static size, loops over `P` and `Q` can then be unrolled, and Kernel Tuner compiles a specialized kernel per configuration anyway. KernelTuner.jl passes all tunable parameters as `Val`s, positionally, in the order of `tune_params`. That's also why `tune_params` is a vector rather than a `Dict`.

### Why does the simple kernel need `BX` and `BY` slots it never uses?

Kernel Tuner passes *all* tunable parameters to the kernel, including the workgroup size that it also uses to launch the kernel. The signature needs a slot for each.

### Which optimization strategy should I use?

Rough guidance:
- Small search space (up to a few hundred configurations that are quick to evaluate): brute force.
- Otherwise: genetic algorithm, dual annealing or differential evolution are robust choices.
- Bayesian optimization pays off when each evaluation is expensive (e.g. with energy measurement), as it chooses fewer, better-informed evaluations.

Always set a limit, and remember that strategies are stochastic. Benchmarking studies of Kernel Tuner's strategies (e.g. Schoonhoven et al., 2022) compare them across many kernels and GPUs.

### What's the difference between restrictions and the configurations Kernel Tuner skips by itself?

Restrictions are evaluated before tuning and shrink the search space, so these configurations are never compiled. Kernel Tuner additionally skips configurations that exceed the device's maximum workgroup size, or that fail to compile or launch; with `verbose=true` it reports these. Configurations that fail are recorded as failed, but still cost compile time.

### On Apple GPUs, why do some large workgroups fail even below 1024 work-items?

On Metal, the maximum number of threads per threadgroup depends on the compiled kernel (`maxTotalThreadsPerThreadgroup`). Kernels with many registers per thread, such as large tiles, can get a lower limit than the device maximum, so they fail to launch and Kernel Tuner skips them.

### How do I use the tuned configuration in my application?

Read `best_config` from the returned `env`, or from the cache file, and launch the kernel with those values: `Val` arguments for the parameters, and the workgroup size and `ndrange` divided by the tile size, like `grid_div` did. For several GPUs, store the best configuration per device name and select it at run time. The Python version of Kernel Tuner has helpers for this in `kernel_tuner.integration` (`store_results`, `create_device_targets`).

### Can Kernel Tuner tune non-Julia kernels?

Yes. Kernel Tuner itself (Python) tunes CUDA, HIP, OpenCL, and C/C++/Fortran code, and supports the same features: strategies, observers, metrics and caching. KernelTuner.jl adds the Julia/KernelAbstractions backend and a Julia interface.

### Can I optimize for several objectives at once?

The tuning itself optimizes one `objective`, but you can measure several quantities (time, energy, error) and analyze the trade-offs afterwards, as in Sections 4.3 and 5. Alternatively, define a metric that combines them, e.g. energy-delay product, and use that as the objective.
