# Compute-side Julia evaluator, one per Slurm job: it runs the command the
# coordinator sends, then reads quit and exits. The HTTP server runs on the laptop.
using Dagger, JSON3, LinearAlgebra, Random
include(joinpath(@__DIR__, "..", "demos", "presentation.jl"))
include(joinpath(@__DIR__, "final_visuals.jl"))
using .FinalVisuals
const USE_MPI = "--mpi" in ARGS
# The job has GPUs: load AMDGPU before MPI initializes its processor inventory,
# so the GPU processors are part of it.
const USE_GPU = "--gpu" in ARGS
USE_GPU && using AMDGPU
if USE_MPI
    using MPI
    Dagger.accelerate!(:mpi)
end
const RANK = USE_MPI ? MPI.Comm_rank(MPI.COMM_WORLD) : 0
const DEMOS = Dict("00_julia"=>false, "01_tasks"=>false, "02_datadeps"=>false, "03_gpu"=>false,
                   "05_multigpu"=>false, "06_multinode"=>true,
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
USE_GPU && Core.eval(WORKSPACE, :(using AMDGPU))
# A frame of the final exercise: a DArray is gathered, a plain array copied.
Core.eval(WORKSPACE, :(__sc26_snapshot(x) = x isa $(Dagger.DArray) ? collect(x) : copy(x)))
# Values created by the latest eval live in a newer world than this loop, so
# isdefined/getglobal can miss them; evaluate the symbol to read its binding.
function workspace_value(name, default = nothing)
    try
        Core.eval(WORKSPACE, name)
    catch err
        err isa UndefVarError ? default : rethrow()
    end
end

# Capture (u, v) snapshots for the time slider without putting logging
# code in the student's template: append a guarded push to the main time loop.
contains_symbol(x, s) = x === s || (x isa Expr && any(a -> contains_symbol(a, s), x.args))

function inject_frame_capture(expr)
    function find_loop(x)
        x isa Expr || return nothing
        if x.head == :for && length(x.args) == 2 && contains_symbol(x.args[1], :nsteps)
            return x
        end
        for arg in x.args
            found = find_loop(arg)
            found === nothing || return found
        end
        return nothing
    end
    loop = find_loop(expr)
    loop === nothing && return false
    # Run 16 more sampling frames than the submitted code asked for. The
    # stride matches the sampler below, so the extra steps are exactly 16
    # additional frames at the same spacing.
    iterator = loop.args[1]
    if iterator isa Expr && iterator.head == :(=) && length(iterator.args) == 2
        range = iterator.args[2]
        if range isa Expr && range.head == :call && range.args[1] == :(:)
            upper = range.args[end]
            if contains_symbol(upper, :nsteps)
                range.args[end] = :(($upper) + 16 * max(1, ($upper) ÷ 24))
            end
        end
    end
    body = loop.args[2]
    if !(body isa Expr && body.head == :block)
        body = Expr(:block, body)
        loop.args[2] = body
    end
    push!(body.args, quote
        try
            global __sc26_tick, __sc26_stride
            __sc26_tick += 1
            if __sc26_tick == 1
                __sc26_stride = max(1, nsteps ÷ 24)
            end
            if __sc26_tick > 1 && (__sc26_tick - 1) % __sc26_stride == 0
                push!(__sc26_frames, (; u = __sc26_snapshot(u), v = __sc26_snapshot(v)))
            end
        catch
        end
    end)
    return true
end
RANK == 0 && DemoPresentation.emit("worker_ready"; mode=USE_MPI ? "mpi" : "single",
                                   job=get(ENV, "SLURM_JOB_ID", nothing), node=gethostname())

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
        elseif command.op == "exercise"
            demo = String(command.demo)
            haskey(DEMOS, demo) || error("Unknown demo: $demo")
            code, setup = String(command.code), String(command.setup)
            # A fresh module per run: the Function defines my_cholesky!, then
            # Setup and Use chooses the scope and calls it, on every MPI rank.
            m = workspace(Symbol("Exercise_", demo))
            old_threads = BLAS.get_num_threads()
            try
                DemoPresentation.present(; title=String(get(command, :label, "Exercise")),
                                         demo=demo, source=code * "\n" * setup,
                                         mpi=USE_MPI ? MPI : nothing) do
                    Base.include_string(m, code, "Function")
                    Base.include_string(m, setup, "Setup and Use")
                    nothing
                end
            finally
                BLAS.set_num_threads(old_threads)
            end
        elseif command.op == "eval"
            code = String(command.code)
            demo = haskey(command, :demo) && command.demo !== nothing ? String(command.demo) : nothing
            # The final exercise runs its Simulation, then its Setup and Use.
            demo == "final" && (code *= "\n\n" * String(command.setup))
            expr = Meta.parseall(code)
            if demo == "final"
                Core.eval(WORKSPACE, :(__sc26_frames = Any[]; __sc26_tick = 0;
                                        __sc26_stride = 1))
                inject_frame_capture(expr)
            end
            DemoPresentation.present(; title="REPL", demo=demo, source=code, mpi=USE_MPI ? MPI : nothing) do
                result = Core.eval(WORKSPACE, expr)
                if RANK == 0 && result !== nothing
                    show(stdout, MIME"text/plain"(), result)
                    println()
                end
            end
            if demo == "final" && RANK == 0
                # Render the selected fields here, outside the logged run, so the
                # plotting code stays out of the student's source and timeline.
                types = String.(get(command, :visuals, String[]))
                try
                    u = workspace_value(:u)
                    v = workspace_value(:v)
                    u isa Dagger.DArray && (u = collect(u))
                    v isa Dagger.DArray && (v = collect(v))
                    (u isa AbstractMatrix && v isa AbstractMatrix) ||
                        error("the run must leave u and v defined")
                    data = FinalVisuals.payload(u, v;
                                                frames = workspace_value(:__sc26_frames),
                                                params = (; F = workspace_value(:F),
                                                            k = workspace_value(:k)))
                    if get(ENV, "SC26_PORTAL", "0") == "1"
                        DemoPresentation.emit("visual"; demo=demo, types=types, data...)
                    else
                        println("u range: ", extrema(u), " v range: ", extrema(v))
                    end
                catch err
                    message = sprint(showerror, err)
                    if get(ENV, "SC26_PORTAL", "0") == "1"
                        DemoPresentation.emit("visual"; demo=demo, types=types, error=message)
                    else
                        println(stderr, "visualization: ", message)
                    end
                end
            end
        else
            error("Unknown command")
        end
        USE_MPI && MPI.Barrier(MPI.COMM_WORLD)
        RANK == 0 && DemoPresentation.emit("done"; ok=true)
    catch err
        if RANK == 0
            # Show the error in the latest world: it may name functions the run defined.
            Base.invokelatest(showerror, stdout, err, catch_backtrace()); println()
            DemoPresentation.emit("done"; ok=false, error=sprint(showerror, err))
        end
        # An MPI exception may leave another rank in a collective; end the job.
        USE_MPI && MPI.Abort(MPI.COMM_WORLD, 1)
    end
    flush(stdout)
end
