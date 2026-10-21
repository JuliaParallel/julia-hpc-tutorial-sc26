# Load once inside a compute-node REPL:
#   include("harness/session.jl")
#   TutorialSession.benchmark(impl="solution", target="gpu")
# Edited implementations are reloaded; unchanged code retains its compiled methods.
module TutorialSession
const INTERACTIVE_SESSION = true
include("run.jl")
include("trace.jl")

function benchmark(; impl="student", target="cpu", sizes=[1024], bs=256,
                     trials=1, gpus="1", strict=false, reload=false, file="", label=get(ENV, "USER", "anon"))
    args = ["--impl=$impl", "--target=$target", "--sizes=$(join(sizes, ','))",
            "--bs=$bs", "--trials=$trials", "--gpus=$gpus", "--label=$label"]
    strict && push!(args, "--strict")
    reload && push!(args, "--reload")
    isempty(file) || push!(args, "--file=$file")
    return main(args)
end

function trace(; path=joinpath(ROOT, "results", "timeline.json"), kwargs...)
    # Run benchmark once without tracing first, so compilation is already warm.
    Dagger.enable_logging!(; all_task_deps=false, taskfuncnames=true)
    code = try
        benchmark(; kwargs...)
    finally
        logs = Dagger.fetch_logs!()
        Dagger.disable_logging!()
        write_trace(logs, path)
    end
    return code
end

function plot(; label=nothing)
    # Fresh module permits repeated plotting without redefining loaded packages.
    m = Module(gensym(:TutorialPlot))
    Base.include(m, joinpath(@__DIR__, "plot.jl"))
    args = label === nothing ? String[] : ["--label=$label"]
    f = Base.invokelatest(getglobal, m, :main)
    return Base.invokelatest(f, args)
end
end
println("Ready: TutorialSession.benchmark(impl=\"solution\"); TutorialSession.plot()")
