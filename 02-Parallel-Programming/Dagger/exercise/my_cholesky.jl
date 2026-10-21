# EXERCISE: parallel tiled Cholesky with Dagger Datadeps.
#
# Start from the serial version in serial_cholesky.jl and make it parallel:
#
#   1. Wrap the whole loop nest in `Dagger.spawn_datadeps() do ... end`
#   2. Put `Dagger.@spawn` in front of each BLAS/LAPACK call
#   3. Tell Dagger how each tile is used:
#        In(tile)    -- the call only reads this tile
#        InOut(tile) -- the call reads AND writes this tile
#
# That's it. Dagger works out which calls can run at the same time, and the
# same function then runs on CPU threads, on the GPU, and across MPI ranks.
#
# Rules of the road:
#   - Spawn the BLAS/LAPACK function itself (e.g. `Dagger.@spawn BLAS.gemm!(...)`),
#     not a helper function that calls it. Dagger swaps in rocBLAS/rocSOLVER
#     when the task lands on an AMD GPU.
#   - No `fetch`/`wait` inside the region.
#   - Under MPI every rank runs this same function: don't branch on the rank.
#
# Check your work:
#   julia --project -t 4 harness/run.jl --impl=student --target=cpu
#
using Dagger
using LinearAlgebra
import LinearAlgebra: BLAS, LAPACK

function my_cholesky!(T::AbstractMatrix)
    mt = size(T, 1)
    # TODO: wrap in Dagger.spawn_datadeps() do ... end
    for k in 1:mt
        # TODO: spawn + annotate
        LAPACK.potrf!('L', T[k, k])

        for m in k+1:mt
            # TODO: spawn + annotate
            BLAS.trsm!('R', 'L', 'T', 'N', 1.0, T[k, k], T[m, k])
        end

        for n in k+1:mt
            # TODO: spawn + annotate
            BLAS.syrk!('L', 'N', -1.0, T[n, k], 1.0, T[n, n])
            for m in n+1:mt
                # TODO: spawn + annotate
                BLAS.gemm!('N', 'T', -1.0, T[m, k], T[n, k], 1.0, T[m, n])
            end
        end
    end
    return T
end
