# Demo 1: tasks. `Dagger.@spawn` returns right away and the work runs on some
# thread. Passing one task's result into another builds a dependency graph.
#
#   julia --project -t 4 demos/01_tasks.jl

isdefined(@__MODULE__, :DemoPresentation) || include(joinpath(@__DIR__, "presentation.jl"))

# source-begin
using Dagger

# Report which thread ran the task (the sleep makes the overlap visible).
whereami() = (sleep(0.1); Threads.threadid())

function run_tasks()
    # (1) Spawn tasks. Passing a task as an argument makes a dependency, so
    #     `c` waits for `a` and `b` with no explicit synchronization.
    a = Dagger.@spawn rand(2000, 2000)
    b = Dagger.@spawn rand(2000, 2000)
    c = Dagger.@spawn a * b
    d = Dagger.@spawn sum(c)
    println("sum = ", fetch(d))

    # (2) Independent tasks run in parallel across threads. `Dagger.@spawn`
    #     takes a function call, so wrap multi-step work in a function.
    tids = fetch.([Dagger.@spawn whereami() for _ in 1:8])
    println("ran on threads: ", tids)
end
# source-end

function run_demo()
    DemoPresentation.present(run_tasks; title="01 Tasks and dependencies",
                             files=[@__FILE__], demo="01_tasks")
end
if abspath(PROGRAM_FILE) == @__FILE__
    run_demo()
end
