# Exercise 4.1: Output verification

## Solution

```julia
answer = [U_ref, V_ref, fill(nothing, length(arguments) - 2)...]
```

Written out, without `fill`:

```julia
answer = [U_ref, V_ref, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing]
```

**Explanation.** `answer` has one entry per kernel argument, in the same order as `arguments`. The kernel writes its results to the first two arguments, `Uout` and `Vout`, so these get the reference arrays; `nothing` tells Kernel Tuner to skip the inputs and scalar arguments. After each configuration, Kernel Tuner copies the outputs back from the GPU and checks that every value is within `atol` of the answer.

## Answers to the questions

1. **`atol=0`.** Kernel Tuner will most likely stop with a verification error, even though the kernel is correct. The GPU and the CPU do not round exactly the same: the compilers may evaluate the floating-point operations in a different order or fuse a multiply and an add into one instruction, which rounds once instead of twice. The results then differ in the last bits. That's why we allow a small tolerance; it should be well above these rounding differences but far below the size of an actual bug.
2. **Why the simple kernel as reference.** The reference must be independent of what we test. If we computed it with the tiled kernel, we would only check that all configurations agree with *one* configuration of the same code, and a bug in the tiling logic would go unnoticed. The simple kernel is easier to get right and was already validated against the CPU implementation in the KernelAbstractions tutorial. Running it on the CPU also avoids any GPU-specific problem in the reference.
