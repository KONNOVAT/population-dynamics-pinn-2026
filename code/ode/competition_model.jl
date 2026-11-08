using Pkg
Pkg.activate(@__DIR__)

using LinearAlgebra
using OrdinaryDiffEq
using Plots
using Printf

const OUTPUT_DIR = joinpath(@__DIR__, "results", "classical")
mkpath(OUTPUT_DIR)

"""Правая часть классической модели межвидовой конкуренции."""
function competition!(du, u, p, t)
    N1, N2 = u
    r1, r2, K1, K2, α12, α21 = p
    du[1] = r1 * N1 * (1 - (N1 + α12 * N2) / K1)
    du[2] = r2 * N2 * (1 - (N2 + α21 * N1) / K2)
    return nothing
end

"""Классификация режима по двум отдельным пороговым неравенствам."""
function classify_regime(p)
    _, _, K1, K2, α12, α21 = p
    first_threshold = K1 / K2
    second_threshold = K2 / K1
    if α12 < first_threshold && α21 < second_threshold
        return :coexistence
    elseif α12 < first_threshold && α21 > second_threshold
        return :species1_excludes_species2
    elseif α12 > first_threshold && α21 < second_threshold
        return :species2_excludes_species1
    elseif α12 > first_threshold && α21 > second_threshold
        return :bistability
    end
    return :boundary_case
end

"""Все аналитические равновесия; внутренняя точка добавляется только при N1*, N2* > 0."""
function equilibria(p)
    _, _, K1, K2, α12, α21 = p
    points = [
        (name="E0", value=[0.0, 0.0]),
        (name="E1", value=[K1, 0.0]),
        (name="E2", value=[0.0, K2]),
    ]
    denominator = 1 - α12 * α21
    if abs(denominator) > 1e-12
        coexistence = [
            (K1 - α12 * K2) / denominator,
            (K2 - α21 * K1) / denominator,
        ]
        if all(coexistence .> 0)
            push!(points, (name="E*", value=coexistence))
        end
    end
    return points
end

"""Матрица Якоби в заданной точке."""
function jacobian_at(point, p)
    N1, N2 = point
    r1, r2, K1, K2, α12, α21 = p
    return [
        r1 * (1 - (2N1 + α12 * N2) / K1)  -r1 * α12 * N1 / K1
        -r2 * α21 * N2 / K2                    r2 * (1 - (2N2 + α21 * N1) / K2)
    ]
end

function stability_label(point, p; tolerance=1e-8)
    λ = eigvals(jacobian_at(point, p))
    real_parts = real.(λ)
    label = if all(real_parts .< -tolerance)
        "устойчиво"
    elseif all(real_parts .> tolerance)
        "неустойчиво"
    elseif minimum(real_parts) < -tolerance && maximum(real_parts) > tolerance
        "седло"
    else
        "негиперболично"
    end
    return label, λ
end

function expected_stable_names(regime)
    regime == :coexistence && return ["E*"]
    regime == :species1_excludes_species2 && return ["E1"]
    regime == :species2_excludes_species1 && return ["E2"]
    regime == :bistability && return ["E1", "E2"]
    return String[]
end

function nearest_equilibrium(final_state, points, allowed_names)
    candidates = filter(eq -> eq.name in allowed_names, points)
    isempty(candidates) && return (name="—", distance=NaN)
    distances = [norm(final_state .- eq.value) for eq in candidates]
    idx = argmin(distances)
    return (name=candidates[idx].name, distance=distances[idx])
end

const REGIMES = [
    (
        key="coexistence",
        title="Устойчивое сосуществование",
        p=[1.0, 1.0, 100.0, 100.0, 0.5, 0.3],
        initial_conditions=[[10.0, 10.0], [80.0, 10.0], [10.0, 80.0], [70.0, 70.0]],
        tspan=(0.0, 30.0),
    ),
    (
        key="exclusion",
        title="Вытеснение вида 2 видом 1",
        p=[1.0, 1.0, 100.0, 80.0, 0.4, 1.5],
        initial_conditions=[[10.0, 10.0], [20.0, 70.0], [80.0, 10.0], [60.0, 60.0]],
        tspan=(0.0, 30.0),
    ),
    (
        key="bistability",
        title="Бистабильность",
        p=[1.0, 1.0, 100.0, 100.0, 1.5, 1.5],
        initial_conditions=[[80.0, 20.0], [20.0, 80.0], [65.0, 35.0], [35.0, 65.0]],
        tspan=(0.0, 30.0),
    ),
]

summary_path = joinpath(OUTPUT_DIR, "equilibria_summary.csv")
trajectory_path = joinpath(OUTPUT_DIR, "trajectory_summary.csv")
open(summary_path, "w") do io
    println(io, "regime,equilibrium,N1,N2,stability,lambda1,lambda2")
end
open(trajectory_path, "w") do io
    println(io, "regime,initial_N1,initial_N2,final_N1,final_N2,nearest_stable_equilibrium,distance")
end

for reg in REGIMES
    regime_kind = classify_regime(reg.p)
    points = equilibria(reg.p)
    allowed_stable = expected_stable_names(regime_kind)
    println("\n", "="^68)
    println(reg.title, " (", regime_kind, ")")
    println("="^68)

    for eq in points
        stability, λ = stability_label(eq.value, reg.p)
        @printf("%-3s = (%8.4f, %8.4f), %-12s, λ=(%9.5f, %9.5f)\n",
                eq.name, eq.value[1], eq.value[2], stability, real(λ[1]), real(λ[2]))
        open(summary_path, "a") do io
            @printf(io, "%s,%s,%.10f,%.10f,%s,%.10f,%.10f\n",
                    reg.key, eq.name, eq.value[1], eq.value[2], stability, real(λ[1]), real(λ[2]))
        end
    end

    solutions = []
    outcomes = String[]
    for u0 in reg.initial_conditions
        problem = ODEProblem(competition!, u0, reg.tspan, reg.p)
        solution = solve(problem, Tsit5(); abstol=1e-10, reltol=1e-9, saveat=0.05)
        push!(solutions, solution)
        minimum_state = minimum(minimum.(solution.u))
        minimum_state >= -1e-7 || error("Отрицательная численность в режиме $(reg.key): $minimum_state")

        final_state = collect(solution.u[end])
        nearest = nearest_equilibrium(final_state, points, allowed_stable)
        nearest.distance < 1e-3 || error(
            "Траектория $(reg.key), u0=$u0 не сошлась к ожидаемому равновесию: расстояние $(nearest.distance)")
        push!(outcomes, nearest.name)
        @printf("u0=(%5.1f,%5.1f) -> (%9.5f,%9.5f), ближайшее %s\n",
                u0[1], u0[2], final_state[1], final_state[2], nearest.name)
        open(trajectory_path, "a") do io
            @printf(io, "%s,%.6f,%.6f,%.10f,%.10f,%s,%.10g\n",
                    reg.key, u0[1], u0[2], final_state[1], final_state[2], nearest.name, nearest.distance)
        end
    end

    if regime_kind == :bistability
        Set(outcomes) == Set(["E1", "E2"]) || error("Для бистабильности не получены оба исхода E1 и E2")
    else
        length(unique(outcomes)) == 1 || error("Начальные условия дали разные исходы в режиме $(reg.key)")
    end

    time_panels = []
    for (idx, solution) in enumerate(solutions)
        u0 = reg.initial_conditions[idx]
        panel = plot(solution.t, first.(solution.u), label="N₁", color=:blue, linewidth=2,
                     xlabel="Время t", ylabel="Численность",
                     title="N₁(0)=$(Int(u0[1])), N₂(0)=$(Int(u0[2]))", legend=:right)
        plot!(panel, solution.t, last.(solution.u), label="N₂", color=:red, linewidth=2)
        push!(time_panels, panel)
    end
    time_plot = plot(time_panels..., layout=(2, 2), size=(1200, 800), dpi=180,
                     plot_title=reg.title, margin=4Plots.mm)
    savefig(time_plot, joinpath(OUTPUT_DIR, "$(reg.key)_time_series.png"))

    phase_plot = plot(xlabel="N₁", ylabel="N₂", title="Фазовый портрет: $(reg.title)",
                      xlim=(-5, 110), ylim=(-5, 110), xticks=0:25:100, yticks=0:25:100,
                      aspect_ratio=:equal,
                      legend=:outerright, size=(900, 700), dpi=180)
    colors = [:blue, :red, :green, :purple]
    for (idx, solution) in enumerate(solutions)
        u0 = reg.initial_conditions[idx]
        plot!(phase_plot, first.(solution.u), last.(solution.u), color=colors[idx], linewidth=2,
              label="u₀=($(Int(u0[1])), $(Int(u0[2])))")
        scatter!(phase_plot, [u0[1]], [u0[2]], color=colors[idx], marker=:circle, label="")
    end
    if regime_kind == :bistability && reg.p[3] == reg.p[4] && reg.p[5] == reg.p[6]
        plot!(phase_plot, 0:100, 0:100, color=:gray, linestyle=:dash, linewidth=1.5,
              label="сепаратриса N₁=N₂")
    end
    for eq in points
        stability, _ = stability_label(eq.value, reg.p)
        marker_shape = stability == "устойчиво" ? :star5 : stability == "седло" ? :diamond : :xcross
        scatter!(phase_plot, [eq.value[1]], [eq.value[2]], marker=marker_shape, markersize=8,
                 color=:black, label="$(eq.name): $(stability)")
    end
    savefig(phase_plot, joinpath(OUTPUT_DIR, "$(reg.key)_phase_portrait.png"))
end

println("\nВсе аналитические и численные проверки пройдены.")
println("Результаты: $OUTPUT_DIR")
