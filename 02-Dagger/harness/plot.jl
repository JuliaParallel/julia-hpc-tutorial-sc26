# Plot benchmark results written by harness/run.jl, right in the terminal.
#
#   julia --project harness/plot.jl                      # everything in results/results.csv
#   julia --project harness/plot.jl --label=student      # just one person's runs
#   julia --project harness/plot.jl path/to/other.csv    # a different results file
#
# One line per configuration (impl, target, ranks x threads, tile size),
# showing the best passing trial at each matrix size.

using DelimitedFiles, Printf, UnicodePlots

function main(args)
    path = joinpath(dirname(@__DIR__), "results", "results.csv")
    label = nothing
    for a in args
        if startswith(a, "--label=")
            label = a[9:end]
        else
            path = a
        end
    end
    isfile(path) || (println("No results yet at $path -- run harness/run.jl first."); return)

    data, header = readdlm(path, ','; header=true)
    col = Dict(strip(h) => i for (i, h) in enumerate(vec(header)))
    rows = [data[i, :] for i in 1:size(data, 1)]
    label === nothing || filter!(r -> string(r[col["label"]]) == label, rows)
    filter!(r -> string(r[col["pass"]]) == "true", rows)
    isempty(rows) && (println("No passing results to plot."); return)

    # series name => (n => best GFLOP/s)
    series = Dict{String,Dict{Int,Float64}}()
    for r in rows
        name = @sprintf("%s %s %dx%d bs=%d", r[col["impl"]], r[col["target"]],
                        r[col["nranks"]], r[col["nthreads"]], r[col["bs"]])
        best = get!(series, name, Dict{Int,Float64}())
        n = Int(r[col["n"]])
        best[n] = max(get(best, n, 0.0), Float64(r[col["gflops"]]))
    end

    names = sort!(collect(keys(series)))
    # UnicodePlots sizes the axes from the first series only, so fix them up front
    all_n = [n for d in values(series) for n in keys(d)]
    ymax = maximum(v for d in values(series) for v in values(d))
    lims = (xlim=(minimum(all_n), maximum(all_n)), ylim=(0, ceil(Int, 1.05ymax)))
    plt = nothing
    for name in names
        ns = sort!(collect(keys(series[name])))
        gf = [series[name][n] for n in ns]
        if plt === nothing
            plt = lineplot(ns, gf; name, lims..., xscale=:log2, xlabel="matrix size N",
                           ylabel="GFLOP/s", title="Tiled Cholesky (best passing trial)",
                           width=60, height=18)
        else
            lineplot!(plt, ns, gf; name)
        end
        length(ns) == 1 && scatterplot!(plt, ns, gf)
    end
    println(plt)

    println("\n", rpad("configuration", 40), "       N     GFLOP/s")
    for name in names, n in sort!(collect(keys(series[name])))
        @printf("%-40s %7d %11.1f\n", name, n, series[name][n])
    end
end

main(ARGS)
