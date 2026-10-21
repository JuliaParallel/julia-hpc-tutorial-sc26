# Demo 4: the SAME Datadeps code across MPI ranks. Every rank runs this whole
# script (SPMD); Dagger splits the tiles across ranks and moves data for you.
#   bash frontier/launch.sh 2 demos/04_mpi.jl  (Odo: two ranks on one node)
#   julia --project -e 'using MPI; run(`$(MPI.mpiexec()) -n 2 $(Base.julia_cmd()) --project -t 2 demos/04_mpi.jl`)'
using Dagger, MPI, Random
Dagger.accelerate!(:mpi)
const rank = MPI.Comm_rank(MPI.COMM_WORLD)

double!(X) = (X .*= 2; write(stdout, "  tile doubled on rank $rank\n"); nothing)

# Rule #1 of Dagger+MPI: every rank must make the same decisions -- so build
# the same data everywhere (same seed), and never branch on `rank` for work.
A = rand(Xoshiro(1), 4096, 4096)
DA = DArray(A, Blocks(1024, 1024))
Dagger.spawn_datadeps() do
    for (i, tile) in enumerate(DA.chunks)
        # Make rank participation visible even for this small demonstration.
        placement = Dagger.scope(mpi_rank=mod(i - 1, MPI.Comm_size(MPI.COMM_WORLD)))
        Dagger.@spawn scope=placement double!(InOut(tile))
    end
end
result = collect(DA)                             # gathered on every rank
@assert result ≈ 2 .* A
rank == 0 && println("correct across $(MPI.Comm_size(MPI.COMM_WORLD)) ranks: ", result ≈ 2 .* A)
