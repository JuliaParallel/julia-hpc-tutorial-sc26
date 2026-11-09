# INSTRUCTOR SOLUTION -- parallel tiled Cholesky with Dagger Datadeps.
# Diff this against exercise/serial_cholesky.jl: the loops are identical;
# only `spawn_datadeps`, `Dagger.@spawn`, and In/InOut were added.
#
# The four tile operations are `const` aliases of the BLAS/LAPACK functions.
# Keep them as aliases: Dagger selects the rocBLAS/rocSOLVER version only when
# the spawned function *is* the BLAS/LAPACK function.

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
    Dagger.spawn_datadeps() do
        for k in 1:mt
            # (1) Factor the diagonal tile: T[k,k] <- chol(T[k,k]).   (LAPACK.potrf!)
            Dagger.@spawn factor_diagonal_tile!('L', InOut(T[k, k]))

            # (2) Solve the panel below the diagonal:
            #     T[m,k] <- T[m,k] * inv(T[k,k]').   (BLAS.trsm!)
            for m in k+1:mt
                Dagger.@spawn solve_panel_tile!('R', 'L', 'T', 'N', 1.0, In(T[k, k]), InOut(T[m, k]))
            end

            # (3) Update the trailing matrix with the new panel.
            for n in k+1:mt
                # Diagonal tile: T[n,n] <- T[n,n] - T[n,k] * T[n,k]'.   (BLAS.syrk!)
                Dagger.@spawn update_diagonal_tile!('L', 'N', -1.0, In(T[n, k]), 1.0, InOut(T[n, n]))
                for m in n+1:mt
                    # Off-diagonal tile: T[m,n] <- T[m,n] - T[m,k] * T[n,k]'.
                    #     (BLAS.gemm!)
                    Dagger.@spawn update_offdiagonal_tile!('N', 'T', -1.0, In(T[m, k]), In(T[n, k]), 1.0, InOut(T[m, n]))
                end
            end
        end
    end
    return T
end
