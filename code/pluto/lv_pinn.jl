### A Pluto.jl notebook ###
# v1.1.0

using Markdown
using InteractiveUtils

# ╔═╡ f9e61639-2edc-45d9-9e96-ff0858710c60
begin
    import Pkg
    Pkg.activate(@__DIR__)
    ENV["GKSwstype"] = "100"
    using NeuralPDE, Lux, Plots, OrdinaryDiffEq, Random, OptimizationOptimJL, LinearAlgebra
    LinearAlgebra.BLAS.set_num_threads(1)
end

# ╔═╡ ce0ddd87-7356-4781-ad05-d103d25255b4
md"""
# Модель Лотки — Вольтерры: численный метод и PINN
Сравнение Tsit5 и сети Lux с оптимизатором BFGS на интервале **0–8**.
PINN обучается последовательно на 16 отрезках длиной 0.5. Начальное состояние
каждого отрезка — результат предыдущей сети; данные Tsit5 при обучении не используются.
Это изменение исходной схемы обучения, в которой одна сеть на всём интервале
сходилась к плохому приближению.
Графики сохраняются в `image/lv_pinn` рядом с этим блокнотом.
Окружение: `Project.toml` и `Manifest.toml` в этой же папке.
"""

# ╔═╡ 0bf55a84-d2a6-4dc4-8aa3-1650c8784b94
begin
    function lotka_volterra(u, p, t)
        α, β, γ, δ = p
        x, y = u
        return [(α - β*y)*x, (δ*x - γ)*y]
    end
    u0 = [1.0, 1.0]
    p = [1.5, 1.0, 3.0, 1.0]
    tspan = (0.0, 8.0)
    prob = ODEProblem(lotka_volterra, u0, tspan, p)
    dt = 0.01
    results_dir = mkpath(joinpath(@__DIR__, "image", "lv_pinn"))
end

# ╔═╡ bc3fddfd-6233-4b20-8151-179a7022838b
solution = solve(prob, Tsit5(); saveat=dt, reltol=1e-9, abstol=1e-11)

# ╔═╡ ac48093b-03e4-4313-a871-a7af3dc475b2
begin
    numeric_phase = plot(solution; idxs=(1,2), dpi=300, color=:black,
        legend=false, xlabel="Prey", ylabel="Predator", title="Numeric phase portrait")
    savefig(numeric_phase, joinpath(results_dir, "num_faz.png"))
    numeric_phase
end

# ╔═╡ 8337ac14-5bf0-4038-a360-917f3081a0da
begin
    numeric_time = plot(solution; dpi=300, xlabel="Time", ylabel="Population size",
        linestyle=[:solid :dashdotdot], color=:black, label=["Prey" "Predator"])
    savefig(numeric_time, joinpath(results_dir, "num.png"))
    numeric_time
end

# ╔═╡ e8da3e18-a5ab-4afc-afd2-5ae78e3bc972
begin
    rng = Random.Xoshiro(42)
    chain = Lux.Chain(Lux.Dense(1 => 16, tanh), Lux.Dense(16 => 16, tanh), Lux.Dense(16 => 2))
    ps, st = Lux.setup(rng, chain) |> Lux.f64
    opt = OptimizationOptimJL.BFGS()
    alg = NeuralPDE.NNODE(chain, opt, ps)
end

# ╔═╡ 013f759d-7bd3-4dff-89ff-6a7845c23dc4
begin
    function solve_pinn_segments(problem, network, parameters, optimizer; window=0.5, saveat=0.01)
        state = copy(problem.u0)
        times = [first(problem.tspan)]
        states = [copy(state)]
        segment_solutions = Any[]
        segment_losses = Float64[]
        for start in first(problem.tspan):window:(last(problem.tspan)-window)
            # Уравнение автономно: на каждом отрезке используем локальное время от нуля.
            local_problem = ODEProblem(problem.f, state, (0.0, window), problem.p)
            local_alg = NeuralPDE.NNODE(network, optimizer, deepcopy(parameters))
            local_solution = solve(local_problem, local_alg; verbose=false,
                abstol=1e-10, maxiters=1000, saveat=saveat)
            append!(times, start .+ local_solution.t[2:end])
            append!(states, [copy(u) for u in local_solution.u[2:end]])
            push!(segment_solutions, local_solution)
            push!(segment_losses, local_solution.resid)
            state = copy(last(local_solution.u))
        end
        return OrdinaryDiffEq.SciMLBase.build_solution(problem, nothing, times, states;
            dense=false, retcode=OrdinaryDiffEq.ReturnCode.Success,
            original=segment_solutions, resid=maximum(segment_losses))
    end
    sol = solve_pinn_segments(prob, chain, ps, opt; saveat=dt)
end

# ╔═╡ a5fbd535-9a55-480d-b767-3b3f77db1ac0
begin
    neural_phase = plot(sol; idxs=(1,2), dpi=300, color=:black,
        legend=false, xlabel="Prey", ylabel="Predator", title="PINN phase portrait")
    savefig(neural_phase, joinpath(results_dir, "neural_faz.png"))
    neural_phase
end

# ╔═╡ d8004da7-38f7-4547-8b85-0c1784b87059
begin
    neural_time = plot(sol; dpi=300, xlabel="Time", ylabel="Population size",
        linestyle=[:solid :dashdotdot], color=:black, label=["Prey" "Predator"])
    savefig(neural_time, joinpath(results_dir, "neural.png"))
    neural_time
end

# ╔═╡ df893307-aeca-4be2-8b28-ca23078318a3
function lv_invariant(u, parameters)
    α, β, γ, δ = parameters
    x, y = u
    return α*log(y) - β*y + γ*log(x) - δ*x
end

# ╔═╡ 6a2b30f3-32ff-4bc7-a543-306d9bcb99dc
begin
    invariant_initial = lv_invariant(u0, p)
    reference_samples = [solution(t) for t in sol.t]
    lv_num_error = [(invariant_initial-lv_invariant(u,p))^2 for u in reference_samples]
    # Инвариант с логарифмом определён только для положительных популяций.
    nonpositive_predictions = count(u -> any(v -> v <= 0, u), sol.u)
    lv_neural_error = [all(v -> v > 0, u) ? (invariant_initial-lv_invariant(u,p))^2 : NaN for u in sol.u]
    relative_l2_error = norm(Array(sol) - reduce(hcat, reference_samples)) / norm(reduce(hcat, reference_samples))
end

# ╔═╡ 824b4fda-627f-42c3-a267-9c54606f42f0
begin
    invariant_plot = plot(sol.t, lv_num_error; label="Numeric method", xlabel="Time",
        ylabel="Squared invariant drift", color=:black, dpi=300)
    plot!(invariant_plot, sol.t, lv_neural_error; label="PINN", color=:black, linestyle=:dash)
    savefig(invariant_plot, joinpath(results_dir, "compare_pinnVSnum_error.png"))
    invariant_plot
end

# ╔═╡ f1d5f111-01f7-4acb-986d-93e82f253ad3
begin
    comparison_plot = plot(sol; xlabel="Time", ylabel="Population size", xlims=tspan,
        label=["Prey. PINN" "Predator. PINN"], linestyle=[:dash :solid], color=:black,
        linewidth=2, dpi=300, size=(800,600), title="Population Dynamics Comparison")
    plot!(comparison_plot, solution; label=["Prey. Tsit5" "Predator. Tsit5"],
        linestyle=[:dashdot :dashdotdot], color=:black, linewidth=2)
    savefig(comparison_plot, joinpath(results_dir, "compare_pinnVSnum.png"))
    comparison_plot
end

# ╔═╡ e279f8c6-c3f7-4e7c-9a3f-9865a7d25548
md"""
### Проверка результата
- Численное решение: **$(solution.retcode)**.
- Решение PINN: **$(sol.retcode)**; максимальная функция потерь по отрезкам: **$(sol.resid)**.
- Относительное L2-отклонение PINN от Tsit5: **$(round(relative_l2_error; sigdigits=5))**.
- Точек PINN с неположительной популяцией: **$(nonpositive_predictions)**.

График инварианта показывает нарушение сохраняющейся величины, а не ошибку
относительно точного решения. `Success` подтверждает выполнение алгоритма;
качество приближения следует оценивать по сравнению с Tsit5 и функции потерь.
"""

# ╔═╡ 33796496-f882-463d-b925-b9e330ec0713
begin
    @assert all(isfinite, Array(solution))
    @assert all(isfinite, Array(sol))
    @assert length(sol.t) == 801
    @assert nonpositive_predictions == 0
    @assert relative_l2_error < 0.05
    (numeric_status=solution.retcode, pinn_status=sol.retcode,
        saved_figures=sort(filter(f -> endswith(f, ".png"), readdir(results_dir))))
end

# ╔═╡ Cell order:
# ╠═f9e61639-2edc-45d9-9e96-ff0858710c60
# ╟─ce0ddd87-7356-4781-ad05-d103d25255b4
# ╠═0bf55a84-d2a6-4dc4-8aa3-1650c8784b94
# ╠═bc3fddfd-6233-4b20-8151-179a7022838b
# ╠═ac48093b-03e4-4313-a871-a7af3dc475b2
# ╠═8337ac14-5bf0-4038-a360-917f3081a0da
# ╠═e8da3e18-a5ab-4afc-afd2-5ae78e3bc972
# ╠═013f759d-7bd3-4dff-89ff-6a7845c23dc4
# ╠═a5fbd535-9a55-480d-b767-3b3f77db1ac0
# ╠═d8004da7-38f7-4547-8b85-0c1784b87059
# ╠═df893307-aeca-4be2-8b28-ca23078318a3
# ╠═6a2b30f3-32ff-4bc7-a543-306d9bcb99dc
# ╠═824b4fda-627f-42c3-a267-9c54606f42f0
# ╠═f1d5f111-01f7-4acb-986d-93e82f253ad3
# ╟─e279f8c6-c3f7-4e7c-9a3f-9865a7d25548
# ╠═33796496-f882-463d-b925-b9e330ec0713
