# INSTRUCTOR SOLUTION -- parallel tiled Cholesky with Dagger Datadeps.
# Diff this against exercise/serial_cholesky.jl: the loops are identical;
# only `spawn_datadeps`, `Dagger.@spawn`, and In/InOut were added.
#
# The file has both halves of a console exercise: the Function (my_cholesky!)
# and, at the end, its Setup and Use for the CPU in one process. Run it on its
# own with   julia --project -t 4 solution/my_cholesky.jl
# (the harness, harness/run.jl --impl=solution, loads only the Function).
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

# Other targets change only the scope line (1); the Function stays the same:
#   one GPU:        using AMDGPU;  scope = Dagger.scope(rocm_gpu=1)
#   every GPU:      using AMDGPU;  scope = Dagger.scope(rocm_gpus=:)
#   CPU and GPU:    scope = Dagger.UnionScope(Dagger.scope(worker=1), Dagger.scope(rocm_gpu=1))
#   MPI ranks:      using MPI; Dagger.accelerate!(:mpi)
#                   scope = Dagger.constrain(Dagger.scope(mpi_ranks=:), Dagger.DefaultScope())
#                   (start with srun or mpiexec, one process per rank)

if abspath(PROGRAM_FILE) == @__FILE__
    # Setup and Use: where the tasks run, the test matrix, the call to
    # my_cholesky!, and the check. Run exercise runs the Function first,
    # then this code.

    using Dagger, LinearAlgebra, Random

    BLAS.set_num_threads(1)   # Dagger runs one tile operation per thread

    # (1) Where the tasks run: the CPU threads.
    scope = Dagger.scope(worker=1)

    # (2) A symmetric positive-definite test matrix. It is seeded, so every
    #     MPI rank builds the same one.
    n, bs = 1024, 256
    A = rand(Xoshiro(1234), n, n)
    A = A + A'
    A[diagind(A)] .+= 2n

    # (3) Tile A, factor the tiles in place, and gather the lower factor L.
    L = Dagger.with_options(scope=scope) do
        DA = DArray(A, Blocks(bs, bs))
        my_cholesky!(DA.chunks)
        LowerTriangular(collect(DA))
    end

    # (4) Check A = L * L' with a scaled residual; it passes below 16.
    x = rand(Xoshiro(7), n)
    residual = norm(A * x - L * (L' * x), Inf) / (norm(A, Inf) * norm(x, Inf) * n * eps())
    status = residual < 16 ? "PASS" : "FAIL"
    println("$status: n = $n, tiles of $bs, residual ", round(residual; sigdigits=3))
    residual < 16 || error("my_cholesky! returned a wrong factor")
end
