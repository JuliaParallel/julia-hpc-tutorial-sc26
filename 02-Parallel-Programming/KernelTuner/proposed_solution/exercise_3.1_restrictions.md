# Exercise 3.1: Restrictions

## Solution

```julia
restrictions = ["32 <= block_size_x * block_size_y <= 1024"]
```

Equivalently, with two separate expressions, which must both be true:

```julia
restrictions = ["block_size_x * block_size_y >= 32", "block_size_x * block_size_y <= 1024"]
```

**Explanation.** A restriction is a string with an expression in terms of the tunable parameters. Kernel Tuner removes every configuration for which an expression is false from the search space before tuning starts, so these configurations are never compiled or benchmarked.

## Answers to the questions

1. **The lower bound of 32.** GPUs execute the work-items of a workgroup in groups of a fixed size: 32 on NVIDIA (warps) and Apple (SIMD-groups), 64 on AMD data center GPUs (wavefronts). A workgroup with fewer work-items than that still occupies a whole group, leaving the rest of its lanes idle. Such configurations are practically always slow, so there is no point in trying them.
2. **Remaining workgroup sizes.** Of the 25 workgroup sizes of Exercise 2.1, 21 remain. Removed are 8 × 1, 8 × 2 and 16 × 1 (fewer than 32 work-items) and 128 × 16 (more than 1024). With the 25 tile sizes of Section 3.3, this leaves 21 × 25 = 525 of the 625 configurations, which the cell at the start of Section 3.3 confirms.
