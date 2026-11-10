# Parallel tiled Cholesky with Dagger Datadeps -- YOUR FILE TO EDIT.
#
# Start from exercise/serial_cholesky.jl and make it parallel:
#   1. wrap the loops in `Dagger.spawn_datadeps() do ... end`,
#   2. put `Dagger.@spawn` before each of the four tile operations below,
#   3. wrap each tile argument in `In(...)` (read only) or `InOut(...)`
#      (read and written).
# Keep the algorithm and loop structure exactly as they are.
#
# The four tile operations have descriptive names so the code reads like the
# picture in README.md. They are `const` aliases, not wrapper functions: on the
# GPU, Dagger replaces BLAS/LAPACK with cuBLAS/cuSOLVER only when the spawned
# function *is* the BLAS/LAPACK function. An alias is that same function under
# another name, so it still gets the GPU version; a hand-written wrapper would
# not be swapped, and the GPU run would fail trying to call CPU BLAS on device
# arrays.

using Dagger
using LinearAlgebra
import LinearAlgebra: BLAS, LAPACK

# Factor the diagonal tile in place, using its lower triangle only:
# T[k,k] <- chol(T[k,k]).   (LAPACK.potrf!)
const factor_diagonal_tile! = LAPACK.potrf!

# Solve the panel of tiles below the diagonal for the new factor:
# T[m,k] <- T[m,k] * inv(T[k,k]'), for m > k.   (BLAS.trsm!)
const solve_panel_tile! = BLAS.trsm!

# Symmetric rank-k update of a diagonal tile:
# T[n,n] <- T[n,n] - T[n,k] * T[n,k]', for n > k.   (BLAS.syrk!)
const update_diagonal_tile! = BLAS.syrk!

# Rank-k update of an off-diagonal tile:
# T[m,n] <- T[m,n] - T[m,k] * T[n,k]', for m > n > k.   (BLAS.gemm!)
const update_offdiagonal_tile! = BLAS.gemm!

function my_cholesky!(T::AbstractMatrix)
    mt = size(T, 1)
    for k in 1:mt
        # (1) Factor the diagonal tile: T[k,k] <- chol(T[k,k]).   (LAPACK.potrf!)
        factor_diagonal_tile!('L', T[k, k])

        # (2) Solve the panel below the diagonal: T[m,k] <- T[m,k] * inv(T[k,k]').
        #     (BLAS.trsm!)
        for m in k+1:mt
            solve_panel_tile!('R', 'L', 'T', 'N', 1.0, T[k, k], T[m, k])
        end

        # (3) Update the trailing matrix with the new panel.
        for n in k+1:mt
            # Diagonal tile: T[n,n] <- T[n,n] - T[n,k] * T[n,k]'.   (BLAS.syrk!)
            update_diagonal_tile!('L', 'N', -1.0, T[n, k], 1.0, T[n, n])
            for m in n+1:mt
                # Off-diagonal tile: T[m,n] <- T[m,n] - T[m,k] * T[n,k]'.
                #     (BLAS.gemm!)
                update_offdiagonal_tile!('N', 'T', -1.0, T[m, k], T[n, k], 1.0, T[m, n])
            end
        end
    end
    return T
end
