# Demo 4: the SAME Datadeps code across MPI ranks. Every rank runs this whole
# script (SPMD); Dagger splits the tiles across ranks and moves data for you.
#   srun -n 2 julia --project -t 2 demos/04_mpi.jl                 (Odo)
#   julia --project -e 'using MPI; run(`$(MPI.mpiexec()) -n 2 $(Base.julia_cmd()) --project -t 2 demos/04_mpi.jl`)'
using Dagger, MPI, Random
Dagger.accelerate!(:mpi)
const rank = MPI.Comm_rank(MPI.COMM_WORLD)

double!(X) = (X .*= 2; Core.println("  tile doubled on rank $rank"); nothing)

# Rule #1 of Dagger+MPI: every rank must make the same decisions -- so build
# the same data everywhere (same seed), and never branch on `rank` for work.
A = rand(Xoshiro(1), 4096, 4096)
DA = DArray(A, Blocks(2048, 2048))
Dagger.spawn_datadeps() do
    for tile in DA.chunks
        Dagger.@spawn double!(InOut(tile))       # unchanged from demo 2
    end
end
result = collect(DA)                             # gathered on every rank
rank == 0 && println("correct across $(MPI.Comm_size(MPI.COMM_WORLD)) ranks: ", result ≈ 2 .* A)
