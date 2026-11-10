import Pkg
Pkg.activate(@__DIR__)

include(joinpath(@__DIR__, "competition_pinn.jl"))

using Dates

results_dir = mkpath(joinpath(@__DIR__, "results", "competition"))

function csv_field(value)
    text = replace(string(value), '"' => "\"\"")
    return string('"', text, '"')
end

attempts_path = joinpath(results_dir, "training_attempts.csv")
open(attempts_path, "w") do io
    println(io, "scenario,start_n1,start_n2,attempt,scheme,max_scaled_error,relative_l2,passed,seconds,error")
    for (i, item) in enumerate(competition_cases), (j, initial) in enumerate(item.starts)
        result = pinn_runs[i][j]
        for (attempt_index, attempt) in enumerate(result.attempts)
            values = (
                item.title,
                initial[1],
                initial[2],
                attempt_index,
                attempt.scheme,
                attempt.max_scaled_error,
                attempt.relative_l2,
                attempt.passed,
                attempt.seconds,
                attempt.error,
            )
            println(io, join(csv_field.(values), ','))
        end
    end
end

summary_path = joinpath(results_dir, "run_summary.txt")
open(summary_path, "w") do io
    println(io, "Время запуска: ", Dates.format(now(), dateformat"yyyy-mm-dd HH:MM:SS"))
    println(io, "Julia: ", VERSION)
    println(io, "Pluto notebook: competition_pinn.jl")
    println(io, "NeuralPDE: ", pkgversion(NeuralPDE))
    println(io, "Lux: ", pkgversion(Lux))
    println(io, "OrdinaryDiffEq: ", pkgversion(OrdinaryDiffEq))
    println(io, "OptimizationOptimJL: ", pkgversion(OptimizationOptimJL))
    println(io, "Сценариев: ", competition_summary.scenarios)
    println(io, "Траекторий: ", competition_summary.trajectories)
    println(io, "Максимальная ошибка, нормированная на K: ", competition_summary.max_scaled_error)
    println(io, "Максимальная относительная L2-ошибка: ", competition_summary.max_relative_l2)
    println(io, "Повторно обученных траекторий: ", competition_summary.retrained_trajectories)
    println(io, "Графики: ", join(competition_summary.saved_figures, ", "))
end

println("Проверка завершена.")
println("Метрики: ", attempts_path)
println("Сводка: ", summary_path)
println(competition_summary)
