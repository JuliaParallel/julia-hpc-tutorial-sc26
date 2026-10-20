module Ex_stats

export min_max
export min_max_avg

include("orig.jl")

init(N; array_type=Vector) = array_type(round.(rand(N) * 100))

end
