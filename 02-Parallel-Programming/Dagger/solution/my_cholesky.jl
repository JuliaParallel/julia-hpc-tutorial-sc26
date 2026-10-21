# INSTRUCTOR SOLUTION -- parallel tiled Cholesky with Dagger Datadeps.
# Diff this against exercise/serial_cholesky.jl: the loops are identical;
# only `spawn_datadeps`, `Dagger.@spawn`, and In/InOut were added.

using Dagger
using LinearAlgebra
import LinearAlgebra: BLAS, LAPACK

function my_cholesky!(T::AbstractMatrix)
    mt = size(T, 1)
    Dagger.spawn_datadeps() do
        for k in 1:mt
            Dagger.@spawn LAPACK.potrf!('L', InOut(T[k, k]))

            for m in k+1:mt
                Dagger.@spawn BLAS.trsm!('R', 'L', 'T', 'N', 1.0, In(T[k, k]), InOut(T[m, k]))
            end

            for n in k+1:mt
                Dagger.@spawn BLAS.syrk!('L', 'N', -1.0, In(T[n, k]), 1.0, InOut(T[n, n]))
                for m in n+1:mt
                    Dagger.@spawn BLAS.gemm!('N', 'T', -1.0, In(T[m, k]), In(T[n, k]), 1.0, InOut(T[m, n]))
                end
            end
        end
    end
    return T
end
