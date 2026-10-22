# Persistent compute-side Julia evaluator. The HTTP server runs on the laptop.
using Dagger, JSON3, LinearAlgebra, Random
include(joinpath(@__DIR__, "..", "demos", "presentation.jl"))
const USE_MPI = "--mpi" in ARGS
if USE_MPI
    # Install GPU processors before MPI initializes its processor inventory,
    # including when the first command happens to be a CPU-only demo.
    using AMDGPU, MPI
    Dagger.accelerate!(:mpi)
end
const RANK = USE_MPI ? MPI.Comm_rank(MPI.COMM_WORLD) : 0
const DEMOS = Dict("01_tasks"=>false, "02_datadeps"=>false, "03_gpu"=>false,
                   "04_mpi"=>true, "05_multigpu"=>false, "06_multinode"=>true,
                   "07_multinode_multigpu"=>true)
const LOADED = Dict{String,Module}()
function workspace(name)
    mod = Module(name)
    Core.eval(mod, :(include(path::AbstractString) = Base.include($mod, path)))
    Core.eval(mod, :(eval(expression) = Core.eval($mod, expression)))
    Core.eval(mod, :(const DemoPresentation = $DemoPresentation))
    return mod
end
const WORKSPACE = workspace(:AttendeeREPL)
Core.eval(WORKSPACE, :(using Dagger, LinearAlgebra, Random))
USE_MPI && Core.eval(WORKSPACE, :(using MPI))
RANK == 0 && DemoPresentation.emit("worker_ready"; mode=USE_MPI ? "mpi" : "single")

while true
    line = RANK == 0 ? (eof(stdin) ? "{\"op\":\"quit\"}" : readline(stdin)) : ""
    USE_MPI && (line = MPI.bcast(line, 0, MPI.COMM_WORLD))
    command = JSON3.read(line)
    command.op == "quit" && break
    ENV["SC26_RUN_ID"] = string(get(command, :id, ""))
    try
        if command.op == "demo"
            name = String(command.demo)
            haskey(DEMOS, name) || error("Unknown demo: $name")
            DEMOS[name] == USE_MPI || error("Switch the worker layout for this demo")
            empty!(ARGS)
            append!(ARGS, String.(get(command, :args, String[])))
            USE_MPI && push!(ARGS, "--mpi")
            mod = get!(LOADED, name) do
                m = workspace(Symbol("Demo_", name))
                Base.include(m, joinpath(@__DIR__, "..", "demos", name * ".jl"))
                m
            end
            entry = Base.invokelatest(getglobal, mod, :run_demo)
            Base.invokelatest(entry)
        elseif command.op == "eval"
            code = String(command.code)
            DemoPresentation.present(; title="REPL", source=code, mpi=USE_MPI ? MPI : nothing) do
                result = Core.eval(WORKSPACE, Meta.parseall(code))
                if RANK == 0 && result !== nothing
                    show(stdout, MIME"text/plain"(), result)
                    println()
                end
            end
        else
            error("Unknown command")
        end
        USE_MPI && MPI.Barrier(MPI.COMM_WORLD)
        RANK == 0 && DemoPresentation.emit("done"; ok=true)
    catch err
        if RANK == 0
            showerror(stdout, err, catch_backtrace()); println()
            DemoPresentation.emit("done"; ok=false, error=sprint(showerror, err))
        end
        # An MPI exception may leave another rank in a collective. Restart the
        # step instead of offering a broken persistent prompt.
        USE_MPI && MPI.Abort(MPI.COMM_WORLD, 1)
    end
    flush(stdout)
end
