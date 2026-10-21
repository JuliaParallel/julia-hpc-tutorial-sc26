# A persistent MPI harness prompt. Rank 0 reads options; every rank receives
# identical options and executes the harness uniformly. No Julia eval here.
using MPI
"--mpi" in ARGS || push!(ARGS, "--mpi")
include("session.jl")
const comm = MPI.COMM_WORLD
const rank = MPI.Comm_rank(comm)

if rank == 0
    println("Warming MPI/GPU code once before the prompt...")
end
warm_target = get(ENV, "SC26_MPI_WARM_TARGET", "gpu")
code = TutorialSession.main(["--impl=solution", "--target=$warm_target",
                             "--sizes=1024", "--bs=256", "--trials=1", "--strict",
                             "--label=mpi-warmup"])
MPI.Allreduce(code, max, comm) == 0 || exit(1)

while true
    command = if rank == 0
        print("\nmpi> "); flush(stdout)
        eof(stdin) ? "quit" : strip(readline(stdin))
    else
        ""
    end
    command = MPI.bcast(command, 0, comm)
    command == "quit" && break
    if command == "plot"
        rank == 0 && TutorialSession.plot()
    elseif isempty(command) || command == "help"
        rank == 0 && println("Enter harness options, e.g. --impl=student --target=gpu --sizes=2048 --bs=512; or plot / quit")
    else
        try
            TutorialSession.main(Base.shell_split(command))
        catch err
            rank == 0 && (showerror(stdout, err); println())
        end
    end
    MPI.Barrier(comm)
end
MPI.Barrier(comm)
exit(0)
