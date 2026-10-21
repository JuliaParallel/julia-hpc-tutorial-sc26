# Export Dagger host task events, keeping CPU/GPU processor names as lanes.
# These are task timings, not a hardware GPU-kernel profiler.
using JSON3

function write_trace(logs, path)
    events = Any[]
    lanes = Dict{Tuple{Int,String},Int}()
    Dagger.logs_event_pairs(logs) do worker, start_idx, finish_idx
        start = logs[worker][:core][start_idx]
        finish = logs[worker][:core][finish_idx]
        start.category in (:compute, :move) || return
        id = logs[worker][:id][start_idx]
        lane = hasproperty(id, :processor) ? string(id.processor) : "data movement"
        tid = get!(lanes, (worker, lane)) do
            next = length(lanes) + 1
            push!(events, Dict("ph" => "M", "name" => "thread_name", "pid" => worker,
                               "tid" => next, "args" => Dict("name" => lane)))
            next
        end
        name = haskey(logs[worker], :taskfuncnames) ?
               string(logs[worker][:taskfuncnames][start_idx]) : string(start.category)
        push!(events, Dict("ph" => "X", "cat" => string(start.category), "name" => name,
                           "pid" => worker, "tid" => tid, "ts" => start.timestamp / 1e3,
                           "dur" => (finish.timestamp - start.timestamp) / 1e3))
    end
    mkpath(dirname(path))
    open(path, "w") do io
        JSON3.write(io, Dict("traceEvents" => events))
    end
    println("Task timeline saved to $path ($(length(lanes)) processor lanes)")
    return path
end
