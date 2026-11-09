# Persistent compute-side Julia evaluator. The HTTP server runs on the laptop.
using Dagger, JSON3, LinearAlgebra, Random
include(joinpath(@__DIR__, "..", "demos", "presentation.jl"))
include(joinpath(@__DIR__, "final_visuals.jl"))
using .FinalVisuals
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
const HARNESS = Dict{Tuple{Bool,Bool},Module}()
function harness_module(use_mpi, use_gpu)
    key = (use_mpi, use_gpu)
    get!(HARNESS, key) do
        m = Module(Symbol("ExerciseHarness_", use_mpi ? "mpi" : "single", "_", use_gpu ? "gpu" : "cpu"))
        Core.eval(m, :(include(path::AbstractString) = Base.include($m, path)))
        saved = copy(ARGS)
        try
            empty!(ARGS)
            use_mpi && push!(ARGS, "--mpi")
            use_gpu && push!(ARGS, "--target=gpu")
            Base.include(m, joinpath(@__DIR__, "..", "harness", "run.jl"))
        finally
            empty!(ARGS)
            append!(ARGS, saved)
        end
        m
    end
end
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
# Values created by the latest eval live in a newer world than this loop, so
# isdefined/getglobal can miss them; evaluate the symbol to read its binding.
function workspace_value(name, default = nothing)
    try
        Core.eval(WORKSPACE, name)
    catch err
        err isa UndefVarError ? default : rethrow()
    end
end

# Capture (omega, psi) snapshots for the time slider without putting logging
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
                push!(__sc26_frames, (; u = copy(u), v = copy(v)))
            end
        catch
        end
    end)
    return true
end
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
        elseif command.op == "exercise"
            demo = String(command.demo)
            haskey(DEMOS, demo) || error("Unknown demo: $demo")
            DEMOS[demo] == USE_MPI || error("Switch the worker layout for this demo")
            target = String(get(command, :target, "cpu"))
            target in ("cpu", "gpu", "mixed") || error("Unknown exercise target: $target")
            strict = Bool(get(command, :strict, false))
            gpus = String(get(command, :gpus, "1"))
            gpus in ("1", "all") || error("Unknown exercise GPU selection: $gpus")
            code = String(command.code)
            job = get(ENV, "SLURM_JOB_ID", "local")
            path = joinpath(@__DIR__, "..", "results", "exercise-$job-$demo.jl")
            if RANK == 0
                mkpath(dirname(path))
                write(path, code)
            end
            USE_MPI && MPI.Barrier(MPI.COMM_WORLD)
            args = ["--impl=student", "--file=$path", "--target=$target",
                    "--sizes=$(String(get(command, :sizes, "1024")))",
                    "--bs=$(Int(get(command, :bs, 256)))",
                    "--trials=$(Int(get(command, :trials, 1)))"]
            USE_MPI && push!(args, "--mpi")
            strict && push!(args, "--strict")
            gpus == "all" && push!(args, "--gpus=all")
            harness = harness_module(USE_MPI, target != "cpu")
            old_threads = BLAS.get_num_threads()
            try
                DemoPresentation.present(; title="Exercise · $(uppercase(target))", demo=demo,
                                         source=code, mpi=USE_MPI ? MPI : nothing) do
                    result = Base.invokelatest(getglobal(harness, :main), args)
                    result == 0 || error("Exercise run failed; see the output above")
                end
            finally
                BLAS.set_num_threads(old_threads)
            end
        elseif command.op == "eval"
            code = String(command.code)
            demo = haskey(command, :demo) && command.demo !== nothing ? String(command.demo) : nothing
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
            showerror(stdout, err, catch_backtrace()); println()
            DemoPresentation.emit("done"; ok=false, error=sprint(showerror, err))
        end
        # An MPI exception may leave another rank in a collective. Restart the
        # step instead of offering a broken persistent prompt.
        USE_MPI && MPI.Abort(MPI.COMM_WORLD, 1)
    end
    flush(stdout)
end
