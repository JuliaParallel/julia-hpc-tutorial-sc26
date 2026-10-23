# Presentation helpers, kept outside the algorithms shown to attendees.
module DemoPresentation
using Dagger, UnicodePlots, JSON3, Dates, Highlights

# Tree-sitter Julia highlighting, shared by the terminal and web console.
# Gogh's first Dracula color is nearly black; lift comments for our dark panels.
const SOURCE_THEME = Highlights.Theme("Dracula"; colors=Dict(1=>"#8b9ca8"))

const PREFIX = "@@SC26@@"
emit(kind; kwargs...) = if get(ENV, "SC26_PORTAL", "0") == "1"
    println(PREFIX, JSON3.write((; kind, run=get(ENV, "SC26_RUN_ID", ""), kwargs...)))
    flush(stdout)
end

"Extract the marked algorithm excerpts from the actual file being executed."
function source_excerpt(path)
    parts = String[]
    block = String[]
    active = false
    for line in eachline(path)
        if strip(line) == "# source-begin"
            active && error("Nested source marker in $path")
            active = true
            empty!(block)
        elseif strip(line) == "# source-end"
            active || error("Unmatched source marker in $path")
            active = false
            nonblank = filter(l -> !isempty(strip(l)), block)
            indent = isempty(nonblank) ? 0 : minimum(length(l) - length(lstrip(l)) for l in nonblank)
            push!(parts, join([isempty(strip(l)) ? "" : l[indent+1:end] for l in block], '\n'))
        elseif active
            push!(block, line)
        end
    end
    active && error("Unclosed source marker in $path")
    isempty(parts) && error("No algorithm source markers in $path")
    return join(parts, "\n\n")
end

highlighted(code) = Highlights.highlight(code, :julia, SOURCE_THEME)

function mpi_processor(proc)
    ext = Base.get_extension(Dagger, :MPIExt)
    return ext !== nothing && proc isa ext.MPIProcessor
end

function memory_endpoint(space, local_rank)
    ext = Base.get_extension(Dagger, :MPIExt)
    if ext !== nothing && space isa ext.MPIMemorySpace
        return (; rank=space.rank, space=Dagger.short_name(space.innerSpace))
    end
    return (; rank=local_rank, space=Dagger.short_name(space))
end

"Convert Dagger's paired compute/move events into a portable timeline."
function timeline_events(logs; rank=0, origin=nothing)
    events = NamedTuple[]
    logs === nothing && return events
    stamps = [e.timestamp for log in values(logs) for e in log[:core]]
    isempty(stamps) && return events
    t0 = origin === nothing ? minimum(stamps) : origin
    # Function names are attached to submission events, not compute events.
    names = Dict{Int,String}()
    uid_to_tid = Dict{UInt,Int}()
    for log in values(logs)
        haskey(log, :taskfuncnames) || continue
        for (i, event) in enumerate(log[:core])
            id = log[:id][i]
            if event.category == :add_thunk && event.kind == :start &&
               hasproperty(id, :thunk_id) && log[:taskfuncnames][i] !== nothing
                names[id.thunk_id] = string(log[:taskfuncnames][i])
            end
            if haskey(log, :taskuidtotid) && log[:taskuidtotid][i] !== nothing
                uid, tid = log[:taskuidtotid][i]
                uid_to_tid[uid] = tid
            end
        end
    end
    # Datadeps records exact source/destination spaces when it creates move!.
    # Join its DTask UID to the scheduler's Thunk ID used by compute events.
    copies = Dict{Int,NamedTuple}()
    for log in values(logs)
        haskey(log, :timeline) || continue
        for (i, event) in enumerate(log[:core])
            event.category == :datadeps_copy && event.kind == :finish || continue
            data = log[:timeline][i]
            data === nothing && continue
            tid = get(uid_to_tid, data.thunk_id, nothing)
            tid === nothing && continue
            copies[tid] = (; source=memory_endpoint(data.from_space, rank),
                             destination=memory_endpoint(data.to_space, rank))
        end
    end
    Dagger.logs_event_pairs(logs) do worker, first, last
        log = logs[worker]
        a, b = log[:core][first], log[:core][last]
        a.category in (:compute, :move) || return
        id = log[:id][first]
        name = hasproperty(id, :thunk_id) ? get(names, id.thunk_id, string(a.category)) : string(a.category)
        proc = hasproperty(id, :processor) ? id.processor : nothing
        owner = mpi_processor(proc) ? proc.rank : rank
        inner = mpi_processor(proc) ? proc.innerProc : proc
        processor = inner === nothing ? "data" : Dagger.short_name(inner)
        category = a.category == :move || name == "move!" ? "move" : "compute"
        # Every MPI rank logs the scheduling wrapper, but only its owner runs
        # the ordinary function body. Do not plot spectators as extra compute.
        category == "compute" && owner != rank && return
        copy_info = hasproperty(id, :thunk_id) ? get(copies, id.thunk_id, nothing) : nothing
        source = copy_info === nothing ? nothing : copy_info.source
        destination = copy_info === nothing ? nothing : copy_info.destination
        if category == "move" && copy_info !== nothing
            rank in (source.rank, destination.rank) || return
        end
        target_rank = destination === nothing ? owner : destination.rank
        lane = if category == "move"
            copy_info === nothing ? "rank $rank / data movement" :
                "transfer $(source.rank) → $(destination.rank) / recorded $rank"
        else
            "rank $owner / $processor"
        end
        push!(events, (; rank, logger_rank=rank, worker, lane, name, category,
                       processor, target_rank, source, destination,
                       start=max(0.0, Float64(Int128(a.timestamp) - Int128(t0)) / 1e9),
                       duration=max(0.0, Float64(Int128(b.timestamp) - Int128(a.timestamp)) / 1e9)))
    end
    sort!(events; by=e -> (e.start, e.lane, e.category))
    return events
end

"UnicodePlots Gantt: each horizontal segment is one logged host task/transfer."
function gantt(events; width=nothing, title="Dagger task timeline")
    isempty(events) && return nothing
    lanes = sort!(unique(e.lane for e in events))
    width = width === nothing ? clamp(displaysize(stdout)[2] - maximum(textwidth, lanes) - 8, 24, 90) : width
    span = max(maximum(e.start + e.duration for e in events), 1e-6)
    height = max(6, 2length(lanes))
    plot = lineplot(; title, xlabel="seconds since logging began", width, height,
                    xlim=(0, span), ylim=(0.5, length(lanes)+0.5), yticks=false)
    for (i, lane) in enumerate(lanes)
        y = length(lanes) - i + 1
        row = clamp(floor(Int, (i - 0.5) * height / length(lanes)) + 1, 1, height)
        UnicodePlots.label!(plot, :l, row, lane)
        for event in events
            event.lane == lane || continue
            stop = min(span, max(event.start + event.duration, event.start + span / (2width)))
            lineplot!(plot, [event.start, stop], [y, y];
                      color=event.category == "compute" ? :cyan : :yellow)
        end
    end
    return plot
end

# Register a backend so the same renderer is useful outside the demo wrapper.
Dagger.render_logs(logs::Dict, ::Val{:unicode_gantt}; kwargs...) = gantt(timeline_events(logs); kwargs...)
function Dagger.show_logs(io::IO, logs::Dict, ::Val{:unicode_gantt}; kwargs...)
    plot = Dagger.render_logs(logs, :unicode_gantt; kwargs...)
    plot === nothing ? println(io, "No completed compute/transfer events.") : show(io, MIME"text/plain"(), plot)
end

"Run work with source display, logging, a terminal Gantt and a JSON artifact."
function present(f; title, files=String[], source=nothing, mpi=nothing, demo=nothing)
    rank = mpi === nothing ? 0 : mpi.Comm_rank(mpi.COMM_WORLD)
    source_records = NamedTuple[]
    if rank == 0
        println("\n━━ $title ━━")
        sources = source === nothing ? [(basename(path), source_excerpt(path)) for path in files] : [("REPL", source)]
        for (name, code) in sources
            ansi = highlighted(code)
            push!(source_records, (; file=name, code, ansi))
            println("\n$name — algorithm source\n")
            println(haskey(ENV, "NO_COLOR") || get(ENV, "SC26_COLOR", "1") == "0" ? code : ansi)
            emit("source"; title, demo, file=name, code, ansi)
        end
        println("\nRunning with Dagger logging enabled…")
        flush(stdout)
    end
    mpi === nothing || mpi.Barrier(mpi.COMM_WORLD)
    Dagger.enable_logging!(; all_task_deps=false, taskfuncnames=true,
                            taskuidtotid=true, timeline=true)
    origin = time_ns()
    try
        return f()
    catch err
        # Do not leave peers hanging in the log gather after a rank-local error.
        if mpi !== nothing
            println(stderr, "Demo failed on MPI rank $rank: ", sprint(showerror, err))
            flush(stderr)
            mpi.Abort(mpi.COMM_WORLD, 1)
        end
        rethrow()
    finally
        logs = try
            Dagger.fetch_logs!()
        finally
            Dagger.disable_logging!()
        end
        events = timeline_events(logs; rank, origin)
        if mpi !== nothing
            gathered = mpi.gather(events, mpi.COMM_WORLD)
            rank == 0 && (events = reduce(vcat, gathered; init=NamedTuple[]))
        end
        if rank == 0
            plot = gantt(events)
            println()
            color = !haskey(ENV, "NO_COLOR") && get(ENV, "SC26_COLOR", "1") != "0"
            plot === nothing ? println("No completed compute/transfer events.") :
                show(IOContext(stdout, :color=>color), MIME"text/plain"(), plot)
            println("\ncyan: compute · yellow: data movement · host task timings, including compilation")
            println("Run again for a warm timeline. GPU lanes are host-side tasks, not kernel profiling.")
            path = joinpath(@__DIR__, "..", "results", "demo-timelines",
                            "$(replace(title, r"[^\w-]+" => "-"))-$(Dates.format(now(), "yyyymmdd-HHMMSS-sss")).json")
            mkpath(dirname(path))
            report = (; title, demo, sources=source_records, events,
                        clock="seconds from each rank's post-barrier logging start")
            open(io -> JSON3.write(io, report), path, "w")
            println("Timeline saved to $path")
            emit("timeline"; report..., path)
        end
    end
end
end
