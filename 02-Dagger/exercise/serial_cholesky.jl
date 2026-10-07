# Serial tiled Cholesky factorization -- the STARTING POINT for the exercise.
#
# This is ordinary, sequential Julia. No Dagger. It factors a symmetric
# positive-definite matrix A into A = L * L' (L lower-triangular), working
# tile-by-tile instead of element-by-element.
#
# `T` is an mt x mt grid of square tiles: T[i, j] is the (i, j)-th tile of A.
# Only the lower triangle (i >= j) is read or written.
#
# Your job (in my_cholesky.jl) is to turn this into a parallel Dagger
# Datadeps program. Hint: you should not need to change the loop structure.

using LinearAlgebra
import LinearAlgebra: BLAS, LAPACK

function serial_cholesky!(T::AbstractMatrix)
    mt = size(T, 1)
    for k in 1:mt
        # (1) POTRF: factor the diagonal tile, T[k,k] <- chol(T[k,k])
        LAPACK.potrf!('L', T[k, k])

        # (2) TRSM: solve for the panel of tiles below the diagonal,
        #     T[m,k] <- T[m,k] * inv(T[k,k]')
        for m in k+1:mt
            BLAS.trsm!('R', 'L', 'T', 'N', 1.0, T[k, k], T[m, k])
        end

        # (3) Update the trailing matrix with the new panel
        for n in k+1:mt
            # SYRK: diagonal tiles, T[n,n] <- T[n,n] - T[n,k] * T[n,k]'
            BLAS.syrk!('L', 'N', -1.0, T[n, k], 1.0, T[n, n])
            for m in n+1:mt
                # GEMM: off-diagonal tiles, T[m,n] <- T[m,n] - T[m,k] * T[n,k]'
                BLAS.gemm!('N', 'T', -1.0, T[m, k], T[n, k], 1.0, T[m, n])
            end
        end
    end
    return T
end

# Quick self-check: `julia --project exercise/serial_cholesky.jl`
if abspath(PROGRAM_FILE) == @__FILE__
    n, bs = 8, 2
    A = rand(n, n); A = A + A'; A[diagind(A)] .+= 2n
    mt = n ÷ bs
    T = [A[(i-1)*bs+1:i*bs, (j-1)*bs+1:j*bs] for i in 1:mt, j in 1:mt]
    serial_cholesky!(T)
    L = LowerTriangular(reduce(vcat, [reduce(hcat, T[i, :]) for i in 1:mt]))
    println("serial_cholesky! correct: ", L * L' ≈ A)
end
