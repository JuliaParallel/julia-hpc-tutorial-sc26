# Start MPI ranks inside one Slurm task with MPICH's own launcher, so the
# ranks need no PMI support from Slurm. andromeda has one node, so every rank
# is a local process.
#   julia --project andromeda/mpirun.jl RANKS COMMAND...
using MPI
ranks = parse(Int, ARGS[1])
command = ARGS[2:end]
# Hydra would use srun to reach other nodes when it sees a Slurm job; fork
# keeps the ranks in this task, with the CPUs and GPUs Slurm gave it.
launcher = `$(MPI.mpiexec()) -launcher fork -n $ranks $command`
process = run(ignorestatus(launcher))
exit(process.exitcode)
