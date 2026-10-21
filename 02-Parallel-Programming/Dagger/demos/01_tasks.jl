# Demo 1: tasks. `Dagger.@spawn` returns right away; the work runs on some
# thread. Passing one task's result into another builds a dependency graph.
#   julia --project -t 4 demos/01_tasks.jl
using Dagger

a = Dagger.@spawn rand(2000, 2000)
b = Dagger.@spawn rand(2000, 2000)
c = Dagger.@spawn a * b          # waits for a and b -- no explicit sync
d = Dagger.@spawn sum(c)
println("sum = ", fetch(d))

# Independent tasks run in parallel across threads
# (`Dagger.@spawn` takes a function call, so wrap multi-step work in a function)
whereami() = (sleep(0.1); Threads.threadid())
tids = fetch.([Dagger.@spawn whereami() for _ in 1:8])
println("ran on threads: ", tids)
