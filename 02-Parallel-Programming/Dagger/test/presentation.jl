using Test, Dagger
include(joinpath(@__DIR__, "..", "demos", "presentation.jl"))
using .DemoPresentation

@testset "Algorithm display and timeline backend" begin
    @test occursin("\e[", DemoPresentation.highlighted("function f(x)\n return x + 1\nend"))
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
    lane_row = only(filter(line -> occursin("rank 1", line), split(rendered, '\n')))
    @test any(c -> '⠁' <= c <= '⣿', split(lane_row, '│')[2])
    @test Dagger.render_logs(logs, :unicode_gantt) !== nothing
end
