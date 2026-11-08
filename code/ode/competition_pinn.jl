# PINN (NNODE) для модели межвидовой конкуренции Лотки-Вольтерры
# Нормализованная форма: u = N/K, τ = r*t
# du1/dτ = u1*(1 - u1 - α12*u2)
# du2/dτ = (r2/r1)*u2*(1 - u2 - α21*u1)

using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

using NeuralPDE, Lux, Plots, OrdinaryDiffEq, Random, Printf
using OptimizationOptimJL
using Optimisers: Adam
import WeightInitializers

const SAVEAT = 0.05
const MAXITERS = parse(Int, get(ENV, "PINN_MAXITERS", "5000"))
const SEED = parse(Int, get(ENV, "PINN_SEED", "42"))
const REGIME_FILTER = get(ENV, "PINN_REGIME", "all")
const TRAIN_T_END = parse(Float64, get(ENV, "PINN_T_END", "5.0"))
const VERBOSE_TRAINING = lowercase(get(ENV, "PINN_VERBOSE", "false")) == "true"
const SAVE_PLOTS = lowercase(get(ENV, "PINN_SAVE_PLOTS", "true")) == "true"
const RUN_LABEL = if MAXITERS == 5000 && REGIME_FILTER == "all"
    "_s$(SEED)"
else
    "_diagnostic_$(REGIME_FILTER)_i$(MAXITERS)_s$(SEED)"
end

println("=== Пакеты загружены ===")
println("NeuralPDE: ", pkgversion(NeuralPDE))
println("Lux: ", pkgversion(Lux))

# --- Нормализованная система конкуренции ---
# u1 = N1/K1, u2 = N2/K2, τ = r1*t
# p = [ρ, α12, α21] где ρ = r2/r1
function competition_norm(u, p, t)
    u1, u2 = u
    ρ, α12, α21 = p
    du1 = u1 * (1 - u1 - α12 * u2)
    du2 = ρ * u2 * (1 - u2 - α21 * u1)
    return [du1, du2]
end

# --- Оригинальная система (для проверки) ---
function competition_orig(u, p, t)
    N1, N2 = u
    r1, r2, K1, K2, α12, α21 = p
    dN1 = r1 * N1 * (1 - (N1 + α12 * N2) / K1)
    dN2 = r2 * N2 * (1 - (N2 + α21 * N1) / K2)
    return [dN1, dN2]
end

# --- Режимы ---
regimes = [
    (key="coexist", name="Сосуществование",
     r1=1.0, r2=1.0, K1=100.0, K2=100.0, α12=0.5, α21=0.3,
     N1_0=10.0, N2_0=10.0, T=20.0),
    (key="exclusion", name="Вытеснение вида 2",
     r1=1.0, r2=1.0, K1=100.0, K2=80.0, α12=0.4, α21=1.5,
     N1_0=10.0, N2_0=10.0, T=20.0),
    (key="bistable", name="Бистабильность",
     r1=1.0, r2=1.0, K1=100.0, K2=100.0, α12=1.5, α21=1.5,
     N1_0=80.0, N2_0=20.0, T=15.0),
    (key="coexist_alt", name="Сосуществование: другое начальное условие",
     r1=1.0, r2=1.0, K1=100.0, K2=100.0, α12=0.5, α21=0.3,
     N1_0=70.0, N2_0=20.0, T=20.0),
    (key="bistable_species2", name="Бистабильность: побеждает вид 2",
     r1=1.0, r2=1.0, K1=100.0, K2=100.0, α12=1.5, α21=1.5,
     N1_0=20.0, N2_0=80.0, T=15.0),
]

selected_regimes = if REGIME_FILTER == "all"
    regimes
else
    filter(reg -> reg.key == REGIME_FILTER, regimes)
end

isempty(selected_regimes) && error(
    "Неизвестный PINN_REGIME=$(REGIME_FILTER). Допустимые ключи: " *
    join((reg.key for reg in regimes), ", ") * ", all."
)

outdir = joinpath(@__DIR__, "results", "pinn")
mkpath(outdir)
metrics_path = joinpath(outdir, "competition_metrics$(RUN_LABEL).csv")

open(metrics_path, "w") do io
    println(io, "regime,seed,maxiters,tsit5_seconds,pinn_seconds,mae_u1,mae_u2,max_u1,max_u2,mae_N1,mae_N2,max_N1,max_N2,negative_points")
end

for reg in selected_regimes
    println("\n" * "="^60)
    println("Режим: $(reg.name)  ($(reg.key))")
    println("="^60)

    # Нормализованные параметры
    ρ = reg.r2 / reg.r1
    α12_norm = reg.α12 * reg.K2 / reg.K1
    α21_norm = reg.α21 * reg.K1 / reg.K2
    u1_0 = reg.N1_0 / reg.K1
    u2_0 = reg.N2_0 / reg.K2
    τ_end = min(reg.r1 * reg.T, TRAIN_T_END)

    println("Нормализация: u1₀=$(round(u1_0,digits=3)), u2₀=$(round(u2_0,digits=3)), τ∈[0,$(τ_end)]")
    println("ρ=$(ρ), α̃₁₂=$(round(α12_norm,digits=3)), α̃₂₁=$(round(α21_norm,digits=3))")

    p_norm = [ρ, α12_norm, α21_norm]
    u0_norm = [u1_0, u2_0]
    tspan_norm = (0.0, τ_end)

    prob = ODEProblem(competition_norm, u0_norm, tspan_norm, p_norm)

    # --- Численное решение ---
    # Отдельный прогрев исключает компиляцию Julia из замера Tsit5.
    solve(prob, Tsit5(); saveat=SAVEAT)
    num_run = @timed solve(prob, Tsit5(); saveat=SAVEAT)
    sol_num = num_run.value
    num_seconds = num_run.time
    println("Tsit5: $(length(sol_num.t)) точек")
    println("Время Tsit5: $(round(num_seconds, digits=4)) с")

    # --- PINN: NNODE + Adam ---
    rng = Random.default_rng()
    Random.seed!(rng, SEED)

    chain = Lux.Chain(
        Lux.Dense(1, 64, gelu),
        Lux.Dense(64, 64, gelu),
        Lux.Dense(64, 64, gelu),
        Lux.Dense(64, 64, gelu),
        Lux.Dense(64, 2)
    )
    ps, st = Lux.setup(rng, chain) |> Lux.f64

    println("Обучение: Adam(0.01), WeightedIntervalTraining, $(MAXITERS) итераций, seed=$(SEED)...")
    opt = Adam(0.01)
    strategy = NeuralPDE.WeightedIntervalTraining([0.7, 0.2, 0.1], 300)
    alg = NeuralPDE.NNODE(chain, opt, ps; strategy=strategy)
    pinn_run = @timed solve(
        prob,
        alg;
        verbose=VERBOSE_TRAINING,
        abstol=1e-10,
        maxiters=MAXITERS,
        saveat=SAVEAT,
    )
    sol_pinn = pinn_run.value
    pinn_seconds = pinn_run.time
    println("PINN: $(length(sol_pinn.t)) точек")
    println("Время PINN: $(round(pinn_seconds, digits=4)) с")

    # --- Денормализация и метрики ---
    u1_num = map(x -> x[1], sol_num.u)
    u2_num = map(x -> x[2], sol_num.u)
    N1_num = u1_num .* reg.K1
    N2_num = u2_num .* reg.K2

    sol_pinn_at = [sol_pinn(t) for t in sol_num.t]
    u1_pinn = map(x -> x[1], sol_pinn_at)
    u2_pinn = map(x -> x[2], sol_pinn_at)
    N1_pinn = u1_pinn .* reg.K1
    N2_pinn = u2_pinn .* reg.K2

    t_real = sol_num.t ./ reg.r1  # обратно к реальному времени

    # Метрики в нормализованном пространстве (0-1)
    mae_u1 = sum(abs.(u1_num .- u1_pinn)) / length(u1_num)
    mae_u2 = sum(abs.(u2_num .- u2_pinn)) / length(u2_num)
    max_err_u1 = maximum(abs.(u1_num .- u1_pinn))
    max_err_u2 = maximum(abs.(u2_num .- u2_pinn))

    # Метрики в реальных единицах
    mae_N1 = mae_u1 * reg.K1
    mae_N2 = mae_u2 * reg.K2
    max_N1 = max_err_u1 * reg.K1
    max_N2 = max_err_u2 * reg.K2

    neg_count = count(u -> any(x -> x < -0.01, u), sol_pinn_at)

    println("\n--- Результаты: $(reg.name) ---")
    println("Норм. MAE   u1: $(round(mae_u1, sigdigits=4)),  u2: $(round(mae_u2, sigdigits=4))")
    println("Норм. Max   u1: $(round(max_err_u1, sigdigits=4)),  u2: $(round(max_err_u2, sigdigits=4))")
    println("Реал. MAE   N1: $(round(mae_N1, sigdigits=4)),  N2: $(round(mae_N2, sigdigits=4))")
    println("Реал. Max   N1: $(round(max_N1, sigdigits=4)),  N2: $(round(max_N2, sigdigits=4))")
    println("Отрицательные (< -0.01): $neg_count / $(length(sol_pinn_at))")

    open(metrics_path, "a") do io
        @printf(
            io,
            "%s,%d,%d,%.6f,%.6f,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%d\n",
            reg.key,
            SEED,
            MAXITERS,
            num_seconds,
            pinn_seconds,
            mae_u1,
            mae_u2,
            max_err_u1,
            max_err_u2,
            mae_N1,
            mae_N2,
            max_N1,
            max_N2,
            neg_count,
        )
    end

    if SAVE_PLOTS
        # --- Графики в реальных единицах ---
        plt = plot(t_real, N1_num, label="N₁ Tsit5", linewidth=2, linestyle=:solid,
                  linecolor=:blue, dpi=200)
        plot!(plt, t_real, N2_num, label="N₂ Tsit5", linewidth=2, linestyle=:solid,
              linecolor=:red)
        plot!(plt, t_real, N1_pinn, label="N₁ PINN", linewidth=2, linestyle=:dash,
              linecolor=:blue)
        plot!(plt, t_real, N2_pinn, label="N₂ PINN", linewidth=2, linestyle=:dash,
              linecolor=:red)
        plot!(plt, title="$(reg.name)", xlabel="Время t", ylabel="Численность N",
              legend=:right)
        solution_plot_path = joinpath(outdir, "competition_$(reg.key)$(RUN_LABEL).png")
        savefig(plt, solution_plot_path)

        # График ошибки
        plt_err = plot(t_real, abs.(N1_num .- N1_pinn), label="|ΔN₁|", linewidth=2, dpi=200)
        plot!(plt_err, t_real, abs.(N2_num .- N2_pinn), label="|ΔN₂|", linewidth=2)
        plot!(plt_err, title="Ошибка: $(reg.name)", xlabel="Время t", ylabel="|Num − PINN|")
        error_plot_path = joinpath(outdir, "competition_$(reg.key)_error$(RUN_LABEL).png")
        savefig(plt_err, error_plot_path)

        println("Графики: $(basename(solution_plot_path)), $(basename(error_plot_path))")
    end
end

println("\n=== Все режимы завершены! ===")
println("Метрики: $metrics_path")
