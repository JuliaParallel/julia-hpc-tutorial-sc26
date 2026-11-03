using Dagger
using LinearAlgebra
import LinearAlgebra: BLAS, LAPACK

function my_cholesky!(T::AbstractMatrix)
    mt = size(T, 1)
    for k in 1:mt
        LAPACK.potrf!('L', T[k, k])

        for m in k+1:mt
            BLAS.trsm!('R', 'L', 'T', 'N', 1.0, T[k, k], T[m, k])
        end

        for n in k+1:mt
            BLAS.syrk!('L', 'N', -1.0, T[n, k], 1.0, T[n, n])
            for m in n+1:mt
                BLAS.gemm!('N', 'T', -1.0, T[m, k], T[n, k], 1.0, T[m, n])
            end
        end
    end
    return T
end
