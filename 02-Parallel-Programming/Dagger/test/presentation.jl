using Test, Dagger
include(joinpath(@__DIR__, "..", "demos", "presentation.jl"))
using .DemoPresentation

@testset "Algorithm display and timeline backend" begin
    @test occursin("\e[", DemoPresentation.highlighted("function f(x)\n return x + 1\nend"))
    sample = "# λ example\nfunction tiled!(C::Matrix{Float64})\n Dagger.@spawn sum(C) + 2.5\n println(\"done\")\nend\n"
    colored = DemoPresentation.highlighted(sample)
    @test replace(colored, r"\e\[[0-9;]*m"=>"") == sample
    @test length(Set(m.match for m in eachmatch(r"\e\[38;2;\d+;\d+;\d+m", colored))) >= 6
    source = DemoPresentation.source_excerpt(joinpath(@__DIR__, "..", "demos", "01_tasks.jl"))
    @test occursin("a = Dagger.@spawn", source)
    @test !occursin("DemoPresentation", source)
    @test !occursin("        a =", source)
    for file in ("01_tasks.jl", "02_datadeps.jl", "03_gpu.jl", "04_mpi.jl", "matrix_product.jl")
        code = DemoPresentation.source_excerpt(joinpath(@__DIR__, "..", "demos", file))
        @test Meta.parse("begin\n" * code * "\nend") isa Expr
    end
    @test DemoPresentation.gantt(NamedTuple[]) === nothing

    # An event across a large absolute clock value must keep its small duration.
    proc = Dagger.ThreadProc(1, 1)
    t = UInt64(10^18)
    logs = Dict(1 => Dict(:core => [
        (;kind=:start, category=:add_thunk, timestamp=t),
        (;kind=:start, category=:compute, timestamp=t),
        (;kind=:finish, category=:compute, timestamp=t+250_000_000),
        (;kind=:start, category=:move, timestamp=t+300_000_000),
        (;kind=:finish, category=:move, timestamp=t+400_000_000)],
        :id => fill((;thunk_id=1, processor=proc), 5),
        :taskfuncnames => ["multiply!", nothing, nothing, nothing, nothing]))
    events = DemoPresentation.timeline_events(logs; rank=1, origin=t-100_000_000)
    @test length(events) == 2
    @test events[1].start ≈ 0.1
    @test events[1].duration ≈ 0.25
    @test events[1].name == "multiply!"
    @test events[2].category == "move"
    @test all(e.rank == 1 for e in events)
    rendered = sprint(show, MIME"text/plain"(), DemoPresentation.gantt(events))
    @test occursin("rank 1", rendered)
    lane_row = only(filter(line -> occursin("rank 1 / W:", line), split(rendered, '\n')))
    @test any(c -> '⠁' <= c <= '⣿', split(lane_row, '│')[2])
    @test Dagger.render_logs(logs, :unicode_gantt) !== nothing
end

using MPI
@testset "MPI ownership and data movement endpoints" begin
    ext = Base.get_extension(Dagger, :MPIExt)
    proc = ext.MPIProcessor(Dagger.ThreadProc(1, 2), MPI.COMM_WORLD, 1)
    from = ext.MPIMemorySpace(Dagger.CPURAMMemorySpace(1), MPI.COMM_WORLD, 0)
    to = ext.MPIMemorySpace(Dagger.CPURAMMemorySpace(1), MPI.COMM_WORLD, 1)
    t = UInt64(10^9)
    function sample(name; copy=false)
        core = [(;kind=:start, category=:add_thunk, timestamp=t),
                (;kind=:start, category=:compute, timestamp=t+10),
                (;kind=:finish, category=:compute, timestamp=t+20),
                (;kind=:finish, category=:datadeps_copy, timestamp=t+30)]
        Dict(1 => Dict(:core=>core, :id=>fill((;thunk_id=3, processor=proc), 4),
            :taskfuncnames=>[name,nothing,nothing,nothing],
            :taskuidtotid=>[UInt(42)=>3,nothing,nothing,nothing],
            :timeline=>[nothing,nothing,nothing,copy ? (;thunk_id=UInt(42),from_space=from,to_space=to) : nothing]))
    end
    @test isempty(DemoPresentation.timeline_events(sample("multiply!");rank=0))
    compute = only(DemoPresentation.timeline_events(sample("multiply!");rank=1))
    @test compute.lane == "rank 1 / W: 1, TID: 2"
    @test compute.category == "compute"
    for rank in (0,1)
        move = only(DemoPresentation.timeline_events(sample("move!";copy=true);rank))
        @test move.category == "move"
        @test move.source.rank == 0
        @test move.destination.rank == 1
        @test move.logger_rank == rank
        @test occursin("0 → 1",move.lane)
    end
    @test isempty(DemoPresentation.timeline_events(sample("move!";copy=true);rank=2))
    unknown = only(DemoPresentation.timeline_events(sample("move!");rank=0))
    @test unknown.source === nothing
    @test unknown.category == "move"
end
