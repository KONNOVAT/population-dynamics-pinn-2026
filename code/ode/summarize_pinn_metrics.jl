# Объединяет результаты повторных PINN-прогонов с разными seed.
# Запуск: julia code/ode/summarize_pinn_metrics.jl

using Statistics
using Printf

const RESULT_DIR = joinpath(@__DIR__, "results", "pinn")
const FILE_RE = r"^competition_metrics_diagnostic_all_i1000_s\d+\.csv$"
const OUTPUT = joinpath(RESULT_DIR, "competition_metrics_summary_i1000.csv")

rows = NamedTuple[]
for path in sort(filter(path -> occursin(FILE_RE, basename(path)), readdir(RESULT_DIR; join=true)))
    for line in Iterators.drop(eachline(path), 1)
        fields = split(line, ',')
        push!(rows, (
            regime=fields[1],
            seed=parse(Int, fields[2]),
            tsit5_seconds=parse(Float64, fields[4]),
            pinn_seconds=parse(Float64, fields[5]),
            mae_N1=parse(Float64, fields[10]),
            mae_N2=parse(Float64, fields[11]),
            max_N1=parse(Float64, fields[12]),
            max_N2=parse(Float64, fields[13]),
            negative_points=parse(Int, fields[14]),
        ))
    end
end

isempty(rows) && error("Не найдены CSV повторных прогонов в $RESULT_DIR")

open(OUTPUT, "w") do io
    println(io, "regime,n_seeds,mean_tsit5_seconds,mean_pinn_seconds,mean_mae_N1,mean_mae_N2,worst_max_N1,worst_max_N2,total_negative_points")
    for regime in ("coexist", "exclusion", "bistable")
        selected = filter(row -> row.regime == regime, rows)
        isempty(selected) && continue
        @printf(
            io,
            "%s,%d,%.9f,%.6f,%.10g,%.10g,%.10g,%.10g,%d\n",
            regime,
            length(selected),
            mean(row.tsit5_seconds for row in selected),
            mean(row.pinn_seconds for row in selected),
            mean(row.mae_N1 for row in selected),
            mean(row.mae_N2 for row in selected),
            maximum(row.max_N1 for row in selected),
            maximum(row.max_N2 for row in selected),
            sum(row.negative_points for row in selected),
        )
    end
end

println("Сводка сохранена: $OUTPUT")
