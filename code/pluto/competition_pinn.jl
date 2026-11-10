### A Pluto.jl notebook ###
# v1.1.0

using Markdown
using InteractiveUtils

# ╔═╡ 74d537f6-8ffb-4076-ad35-804e55694b86
begin
    import Pkg
    Pkg.activate(@__DIR__)
    ENV["GKSwstype"] = "100"
    using OrdinaryDiffEq, NeuralPDE, Lux, OptimizationOptimJL, Random, LinearAlgebra, Printf
    using Plots
    LinearAlgebra.BLAS.set_num_threads(1)
    competition_output = mkpath(joinpath(@__DIR__, "results", "competition", "image"))
end

# ╔═╡ 30ddc5af-6ee4-4fa8-9926-b4b63c40f202
md"""
# Две популяции, конкурирующие за общий ресурс
## Логистический рост и межвидовая конкуренция
Используется модель из предоставленных изображений:
```math
\begin{aligned}
\frac{dN_1}{dt}&=r_1N_1\left(1-\frac{N_1+\alpha_{12}N_2}{K_1}\right),\\
\frac{dN_2}{dt}&=r_2N_2\left(1-\frac{N_2+\alpha_{21}N_1}{K_2}\right).
\end{aligned}
```
Это модель **конкуренции**, в которой оба вида подавляют рост друг друга.
Ресурс учитывается **неявно** через ёмкости среды и коэффициенты конкуренции;
отдельного уравнения для количества ресурса здесь нет.

| Обозначение | Смысл |
|:--|:--|
| ``N_1,N_2`` | численности первого и второго видов |
| ``r_1,r_2>0`` | собственные скорости роста |
| ``K_1,K_2>0`` | ёмкости среды для каждого вида по отдельности |
| ``\alpha_{12}\ge0`` | воздействие одной особи вида 2 на вид 1 |
| ``\alpha_{21}\ge0`` | воздействие одной особи вида 1 на вид 2 |

Например, ``\alpha_{12}=0.5`` означает, что одна особь вида 2 создаёт для
вида 1 такое же конкурентное давление, как половина особи вида 1.
Ни один вид не питается другим. Время условное: если ``r_i`` заданы в год⁻¹, ось времени — годы.
"""

# ╔═╡ e723ea08-d14d-4faf-a1f0-23cd8b8e2624
begin
    function competition_rhs(u, q, t)
        n1, n2 = u
        return [q.r1*n1*(1-(n1+q.a12*n2)/q.K1),
                q.r2*n2*(1-(n2+q.a21*n1)/q.K2)]
    end
    function invasion_rates(q)
        return (g1=q.r1*(1-q.a12*q.K2/q.K1), g2=q.r2*(1-q.a21*q.K1/q.K2))
    end
    function competition_regime(q)
        @assert q.r1 > 0 && q.r2 > 0 && q.K1 > 0 && q.K2 > 0
        @assert q.a12 >= 0 && q.a21 >= 0
        g = invasion_rates(q)
        if min(abs(g.g1),abs(g.g2)) < 1e-10
            return :boundary
        elseif g.g1 > 0 && g.g2 < 0
            return :species1
        elseif g.g1 < 0 && g.g2 > 0
            return :species2
        elseif g.g1 > 0 && g.g2 > 0
            return :coexistence
        else
            return :bistability
        end
    end
    function interior_equilibrium(q)
        den = 1-q.a12*q.a21
        abs(den) < 1e-12 && return nothing
        point = [(q.K1-q.a12*q.K2)/den, (q.K2-q.a21*q.K1)/den]
        return all(point .> 0) ? point : nothing
    end
    function competition_jacobian(n, q)
        n1,n2=n
        return [q.r1*(1-(2n1+q.a12*n2)/q.K1) -q.r1*q.a12*n1/q.K1;
                -q.r2*q.a21*n2/q.K2 q.r2*(1-(2n2+q.a21*n1)/q.K2)]
    end
end

# ╔═╡ e91914bc-e8c0-4464-9452-316d4fcb8c26
md"""
## Четыре исхода: условия, а не только примеры
Проверим, может ли редкий вид расти, когда второй вид уже достиг своей ёмкости:
```math
g_1=r_1\left(1-\frac{\alpha_{12}K_2}{K_1}\right),\qquad
g_2=r_2\left(1-\frac{\alpha_{21}K_1}{K_2}\right).
```
| Знаки | Эквивалентные условия | Исход при положительных начальных численностях |
|:--|:--|:--|
| ``g_1>0,\ g_2<0`` | ``K_1>\alpha_{12}K_2,\ K_2<\alpha_{21}K_1`` | вид 1 вытесняет вид 2; предел ``(K_1,0)`` |
| ``g_1<0,\ g_2>0`` | ``K_1<\alpha_{12}K_2,\ K_2>\alpha_{21}K_1`` | вид 2 вытесняет вид 1; предел ``(0,K_2)`` |
| ``g_1>0,\ g_2>0`` | оба вида могут вторгнуться в популяцию другого | устойчивое сосуществование |
| ``g_1<0,\ g_2<0`` | ни один редкий вид не может вторгнуться | бистабильность: победитель зависит от начальных численностей |

При ``\alpha_{12}\alpha_{21}\ne1`` внутреннее равновесие имеет вид
```math
N_1^*=\frac{K_1-\alpha_{12}K_2}{1-\alpha_{12}\alpha_{21}},\qquad
N_2^*=\frac{K_2-\alpha_{21}K_1}{1-\alpha_{12}\alpha_{21}}.
```
В режиме сосуществования оно устойчиво, в режиме бистабильности — седло.
Условия с равенствами — границы режимов; их не следует автоматически относить к одному из четырёх строгих случаев.

При положительных ``r_i`` скорости роста влияют на время перехода, а при бистабильности
могут менять границу бассейнов притяжения. Знаки ``g_i`` и координаты равновесий
определяются ``K_i`` и ``\alpha_{ij}``.

Четыре режима и смысл коэффициентов согласованы с
[учебным модулем University of Tennessee](https://lgross.utk.edu/LGrossTIEMwebsite/home/gross/public_html/bioed/bealsmodules/competition.html).
Формулы равновесий и матрица Якоби ниже также проверяются непосредственно по уравнениям.
"""

# ╔═╡ 19e6e84a-05b3-47f4-82a2-e7ce132063f9
begin
    base_parameters = (r1=1.0, r2=1.0, K1=100.0, K2=100.0)
    competition_cases = [
        (key=:species1, title="1. Победа вида 1", params=merge(base_parameters,(a12=0.5,a21=1.5)), starts=[[50.0,30.0]]),
        (key=:species2, title="2. Победа вида 2", params=merge(base_parameters,(a12=1.5,a21=0.5)), starts=[[50.0,30.0]]),
        (key=:coexistence, title="3. Сосуществование", params=merge(base_parameters,(a12=0.5,a21=0.3)), starts=[[50.0,30.0]]),
        (key=:bistability, title="4. Бистабильность", params=merge(base_parameters,(a12=1.5,a21=1.5)), starts=[[80.0,20.0],[20.0,80.0]])
    ]
    long_horizon = 60.0
    pinn_horizon = 20.0
    sample_dt = 0.05
end

# ╔═╡ a063f8a3-7e06-439f-ae7a-2b789de2b1cf
begin
    scenario_rows = ["| Сценарий | α₁₂ | α₂₁ | N₁(0), N₂(0) | g₁ | g₂ | Внутреннее равновесие |", "|:--|--:|--:|:--|--:|--:|:--|"]
    for item in competition_cases
        g=invasion_rates(item.params)
        eq=interior_equilibrium(item.params)
        eqtext=eq===nothing ? "нет в положительном квадранте" : string(round.(eq;digits=4))
        initialtext=join(string.(item.starts), "; ")
        push!(scenario_rows,"| $(item.title) | $(item.params.a12) | $(item.params.a21) | $initialtext | $(g.g1) | $(g.g2) | $eqtext |")
    end
    Markdown.parse("### Наборы параметров
Все r₁=r₂=1 и K₁=K₂=100. Для сосуществования взяты коэффициенты 0.5 и 0.3 из примера пользователя.

" * join(scenario_rows,"
"))
end

# ╔═╡ f929c6ec-3e95-45e2-888f-c72ea8db8777
begin
    numerical_runs = map(competition_cases) do item
        map(item.starts) do initial
            problem=ODEProblem(competition_rhs,initial,(0.0,long_horizon),item.params)
            reference=solve(problem,Tsit5();reltol=1e-10,abstol=1e-12,saveat=sample_dt)
            check=solve(problem,Vern9();reltol=1e-12,abstol=1e-13,saveat=sample_dt)
            scale=[item.params.K1,item.params.K2]
            solver_difference=maximum(abs.((Array(reference)-Array(check))./scale))
            @assert successful_retcode(reference) && successful_retcode(check)
            @assert solver_difference < 1e-7
            (reference=reference,solver_difference=solver_difference)
        end
    end
end

# ╔═╡ a448a859-6a30-462b-9a95-d868d33c56b0
begin
    endpoint_rows=["| Сценарий и старт | N₁(60) | N₂(60) | Ожидаемый предел | Проверка |", "|:--|--:|--:|:--|:--|"]
    for (i,item) in enumerate(competition_cases), (j,initial) in enumerate(item.starts)
        tail=last(numerical_runs[i][j].reference.u)
        expected=item.key==:species1 ? [item.params.K1,0.0] : item.key==:species2 ? [0.0,item.params.K2] : item.key==:coexistence ? interior_equilibrium(item.params) : j==1 ? [item.params.K1,0.0] : [0.0,item.params.K2]
        mismatch=maximum(abs.(tail-expected)./[item.params.K1,item.params.K2])
        @assert mismatch < 1e-5
        push!(endpoint_rows,@sprintf("| %s; старт (%.0f, %.0f) | %.6g | %.6g | (%.4f, %.4f) | совпадает |",item.title,initial[1],initial[2],tail[1],tail[2],expected[1],expected[2]))
    end
    Markdown.parse("### Долгосрочный исход: численный расчёт до t=60

" * join(endpoint_rows,"
") * "

Вытеснение означает стремление численности к нулю при t→∞. Малое значение в конечный момент времени не является точным математическим нулём.")
end

# ╔═╡ 4157c631-50a9-4c39-96c4-cb422d97731a
md"""
### График 1. Как меняются численности: четыре долгосрочных исхода
**Что показывает:** численное решение Tsit5 на интервале 0–60 для четырёх наборов коэффициентов.
**Оси:** по горизонтали — время, по вертикали — численность в исходных единицах.
**Обозначения:** синий — вид 1, оранжевый — вид 2. В четвёртой панели сплошные линии
соответствуют старту `(80, 20)`, пунктирные — `(20, 80)`; точные подписи приведены в легенде.
Из-за симметрии выбранного примера победные и проигрывающие кривые двух запусков накладываются.
Победителя определяйте по сочетанию цвета и стиля линии и по таблице конечных численностей выше.
**Как читать:** стремление одного вида к нулю показывает вытеснение;
две положительные предельные численности — сосуществование. В четвёртом режиме смена
старта меняет победителя при тех же коэффициентах.
"""

# ╔═╡ f0d996eb-e665-4e52-8c7f-9e736e2a29ab
begin
    time_panels=map(enumerate(competition_cases)) do (i,item)
        panel=plot(;title=item.title,xlabel="Время",ylabel="Численность",legend=:right,size=(680,420))
        for (j,initial) in enumerate(item.starts)
            suffix=length(item.starts)>1 ? " / старт $(Int.(initial))" : ""
            ref=numerical_runs[i][j].reference
            plot!(panel,ref.t,ref[1,:];color=:royalblue,linestyle=j==1 ? :solid : :dash,linewidth=2,label="Вид 1"*suffix)
            plot!(panel,ref.t,ref[2,:];color=:darkorange,linestyle=j==1 ? :solid : :dash,linewidth=2,label="Вид 2"*suffix)
        end
        panel
    end
    outcomes_figure=plot(time_panels...;layout=(2,2),size=(1360,900),dpi=180,margin=5Plots.mm)
    savefig(outcomes_figure,joinpath(competition_output,"four_outcomes_time.png"))
    outcomes_figure
end

# ╔═╡ a9d55169-689b-40b2-9dd0-81df33992b6f
md"""
### График 2. Фазовые портреты: куда движется система
**Что показывает:** совместное изменение численностей при разных начальных состояниях.
**Оси:** по горизонтали — численность вида 1, по вертикали — численность вида 2;
время на этих осях не отложено. Каждая точка означает состояние двух популяций одновременно.
**Как читать:** следуйте чёрной траектории в направлении серых стрелок, чтобы понять,
к какому равновесию приходит система. Цветные пунктирные линии — изоклины нулевого роста.
Обозначения устойчивости точек и смысл красной границы подробно разобраны сразу после рисунка.
"""

# ╔═╡ 40f6bfe2-6060-4c04-8839-d987371556ac
begin
    function competition_phase_panel(item)
        q=item.params
        panel=plot(;title=item.title,xlabel="N₁",ylabel="N₂",xlims=(0,120),ylims=(0,120),aspect_ratio=:equal,legend=:topright)
        xs=range(0,120;length=241)
        iso1=[(q.K1-x)/q.a12 for x in xs]
        iso2=[q.K2-q.a21*x for x in xs]
        plot!(panel,xs,[0<=y<=120 ? y : NaN for y in iso1];color=:royalblue,linestyle=:dash,label="dN₁/dt = 0")
        plot!(panel,xs,[0<=y<=120 ? y : NaN for y in iso2];color=:darkorange,linestyle=:dash,label="dN₂/dt = 0")
        gx=Float64[];gy=Float64[];dx=Float64[];dy=Float64[]
        for x in 6:12:114,y in 6:12:114
            v=competition_rhs([x,y],q,0.0);m=norm(v)
            if m>1e-10
                push!(gx,x);push!(gy,y);push!(dx,4v[1]/m);push!(dy,4v[2]/m)
            end
        end
        quiver!(panel,gx,gy;quiver=(dx,dy),color=:gray70,label="",linewidth=0.6)
        for start in [[10.0,10.0],[10.0,90.0],[90.0,10.0],[110.0,90.0],[30.0,70.0],[70.0,30.0]]
            tr=solve(ODEProblem(competition_rhs,start,(0.0,60.0),q),Tsit5();reltol=1e-9,abstol=1e-11,saveat=0.1)
            plot!(panel,tr[1,:],tr[2,:];color=:black,alpha=0.65,linewidth=1.2,label="")
        end
        equilibria=[[0.0,0.0],[q.K1,0.0],[0.0,q.K2]]
        inner=interior_equilibrium(q)
        inner!==nothing && push!(equilibria,inner)
        for point in equilibria
            stable=maximum(real.(eigvals(competition_jacobian(point,q)))) < -1e-8
            scatter!(panel,[point[1]],[point[2]];markersize=5,markercolor=stable ? :black : :white,markerstrokecolor=:black,label="")
        end
        if item.key==:bistability
            plot!(panel,[0.0,120.0],[0.0,120.0];color=:crimson,linestyle=:dot,linewidth=2,label="Сепаратриса N₁=N₂")
        end
        return panel
    end
    phase_figure=plot(competition_phase_panel.(competition_cases)...;layout=(2,2),size=(1300,1100),dpi=180,margin=5Plots.mm)
    savefig(phase_figure,joinpath(competition_output,"four_outcomes_phase.png"))
    phase_figure
end

# ╔═╡ cb74579e-b9e1-41ce-bb18-439e9f977b18
md"""
### Как читать фазовые портреты
Синяя и оранжевая пунктирные линии — нетривиальные изоклины нулевого роста.
Серые стрелки показывают направление поля; чёрные кривые — численные траектории.
**Закрашенная точка** — устойчивое равновесие, **незакрашенная** — неустойчивое.
В четвёртом примере красная диагональ разделяет бассейны притяжения.
Диагональ является точной сепаратрисой именно для выбранных симметричных параметров
``r_1=r_2``, ``K_1=K_2``, ``\alpha_{12}=\alpha_{21}``.
На ней траектория стремится к седлу ``(40,40)``; вне неё побеждает один из видов.
Для произвольных параметров граница обычно не является диагональю, и правило
«у кого больше особей в начале, тот всегда победит» неверно в общем случае.
"""

# ╔═╡ 9fb23b76-fbdc-47fa-b338-5a29140662c8
md"""
### График 3. Карта параметров: какой исход задают коэффициенты конкуренции
**Что показывает:** четыре области исходов при фиксированных `K₁=K₂=100` и положительных скоростях роста.
**Оси:** по горизонтали — α₁₂, давление вида 2 на вид 1;
по вертикали — α₂₁, давление вида 1 на вид 2.
**Обозначения:** название каждого исхода написано прямо в соответствующей цветной области;
чёрные точки — четыре выбранных набора параметров.
**Как читать:** найдите точку с нужными коэффициентами и прочитайте название области.
Чёрные пунктирные прямые α₁₂=1 и α₂₁=1 — границы режимов, а не дополнительные случаи.
Карта задаёт режим, но внутри области бистабильности конкретный победитель зависит ещё и от старта.
При других ёмкостях среды границы сдвигаются к α₁₂=K₁/K₂ и α₂₁=K₂/K₁.
"""

# ╔═╡ e9128cf9-4e71-42fa-ac49-244554571682
begin
    axis_alpha=collect(range(0.0,2.0;length=201))
    regime_codes=Dict(:species1=>1,:species2=>2,:coexistence=>3,:bistability=>4,:boundary=>NaN)
    regime_grid=[regime_codes[competition_regime(merge(base_parameters,(a12=x,a21=y)))] for y in axis_alpha,x in axis_alpha]
    regime_map=heatmap(axis_alpha,axis_alpha,regime_grid;clims=(0.5,4.5),
        color=cgrad([:lightblue,:moccasin,:lightgreen,:plum],4;categorical=true),colorbar=false,
        xlabel="α₁₂: давление вида 2 на вид 1",ylabel="α₂₁: давление вида 1 на вид 2",
        title="Карта четырёх режимов при K₁=K₂=100",size=(850,650),dpi=180)
    vline!(regime_map,[1.0];color=:black,linestyle=:dash,label="")
    hline!(regime_map,[1.0];color=:black,linestyle=:dash,label="")
    annotate!(regime_map,[(0.5,1.75,text("Победа вида 1",11)),(1.5,0.25,text("Победа вида 2",11)),
        (0.5,0.25,text("Сосуществование",11)),(1.5,1.75,text("Бистабильность",11))])
    scatter!(regime_map,[c.params.a12 for c in competition_cases],[c.params.a21 for c in competition_cases];color=:black,markersize=4,label="Выбранные примеры")
    savefig(regime_map,joinpath(competition_output,"parameter_regimes.png"))
    regime_map
end

# ╔═╡ 463a104f-12e1-4f64-85ed-a35cf11b1d46
md"""
## PINN: схема обучения и причины её выбора
Численный расчёт до ``t=60`` нужен для проверки долгосрочного исхода.
**PINN сравнивается с Tsit5 на интервале 0–20**, одинаковом для всех пяти запусков
(в четвёртом сценарии — два разных начальных состояния).

В предыдущем блокноте хищник–жертва обучение одной сети сразу на длинном интервале
давало неверную траекторию даже при малой обучающей невязке. Поэтому здесь изначально
используется последовательное обучение на коротких отрезках: состояние в конце
отрезка становится начальным условием следующего. Значения Tsit5 **не входят в функцию потерь и не задают промежуточные
начальные условия**. Эталон используется для проверки точности и решения,
нужна ли повторная попытка с другой схемой обучения.

Чтобы масштабы численностей около 100 не ухудшали обучение и сеть не выдавала
отрицательные популяции, введены переменные
```math
z_i=\log(N_i/K_i),\qquad N_i=K_i e^{z_i}.
```
Для строго положительных начальных численностей это эквивалентная замена:
```math
\dot z_1=r_1\left(1-e^{z_1}-\alpha_{12}\frac{K_2}{K_1}e^{z_2}\right),\qquad
\dot z_2=r_2\left(1-e^{z_2}-\alpha_{21}\frac{K_1}{K_2}e^{z_1}\right).
```
Положительность обеспечивается обратным преобразованием, а не обрезанием отрицательных значений.
При начальной численности ровно ноль логарифмическая схема неприменима; в данных примерах оба вида присутствуют.

**Первая попытка:** окно 2, фиксированная сетка 0.05, BFGS, до 600 итераций на окно.
**Если критерий не выполнен:** окно сокращается до 1, сетка заменяется адаптивным
интегрированием невязки, лимит увеличивается до 1000. Это уменьшает сложность локальной задачи
и риск пропустить ошибку между узлами. Реально выполненные попытки и их результаты
публикуются ниже; успешность не подменяется кодом возврата оптимизатора.

Критерий: ``\max_{i,t}|N_i^{PINN}(t)-N_i^{Tsit5}(t)|/K_i<0.01`` на общей сетке 0–20,
все значения конечны и положительны. Дополнительно выводится относительная L2-ошибка.
Это проверка на выбранной сетке, а не доказанная оценка ошибки между всеми её узлами.
"""

# ╔═╡ 54048d1f-1b90-4ec2-897f-b4ee741a1605
begin
    function competition_log_rhs(z,q,t)
        return [q.r1*(1-exp(z[1])-q.a12*q.K2/q.K1*exp(z[2])),
                q.r2*(1-exp(z[2])-q.a21*q.K1/q.K2*exp(z[1]))]
    end
    function train_competition_once(q,initial;window,grid,maxiters,seed=42)
        scale=[q.K1,q.K2]
        logstate=log.(initial./scale)
        network=Lux.Chain(Lux.Dense(1=>16,tanh),Lux.Dense(16=>16,tanh),Lux.Dense(16=>2))
        parameters,_=Lux.setup(Random.Xoshiro(seed),network) |> Lux.f64
        optimizer=OptimizationOptimJL.BFGS()
        times=[0.0]; populations=[copy(initial)]; losses=Float64[]
        for left in 0.0:window:(pinn_horizon-window)
            problem=ODEProblem(competition_log_rhs,logstate,(0.0,window),q)
            algorithm=grid===nothing ? NeuralPDE.NNODE(network,optimizer,deepcopy(parameters)) : NeuralPDE.NNODE(network,optimizer,deepcopy(parameters);strategy=NeuralPDE.GridTraining(grid))
            local_sol=solve(problem,algorithm;saveat=sample_dt,maxiters=maxiters,abstol=1e-9,verbose=false)
            append!(times,left .+ local_sol.t[2:end])
            append!(populations,[scale.*exp.(z) for z in local_sol.u[2:end]])
            logstate=copy(last(local_sol.u))
            push!(losses,Float64(local_sol.resid))
        end
        physical_problem=ODEProblem(competition_rhs,initial,(0.0,pinn_horizon),q)
        predicted=OrdinaryDiffEq.SciMLBase.build_solution(physical_problem,nothing,times,populations;
            dense=false,retcode=OrdinaryDiffEq.ReturnCode.Success,resid=maximum(losses))
        return predicted
    end
    function train_competition_checked(q,initial,reference)
        schemes=[(name="Окно 2, GridTraining(0.05), 600 итераций",window=2.0,grid=0.05,maxiters=600),
                 (name="Окно 1, адаптивная квадратура, 1000 итераций",window=1.0,grid=nothing,maxiters=1000)]
        attempts=NamedTuple[]
        for scheme in schemes
            start_time=time()
            try
                predicted=train_competition_once(q,initial;window=scheme.window,grid=scheme.grid,maxiters=scheme.maxiters)
                target=reduce(hcat,[reference(t) for t in predicted.t])
                data=Array(predicted)
                max_scaled_error=maximum(abs.((data-target)./[q.K1,q.K2]))
                relative_l2=norm(data-target)/norm(target)
                positive=all(isfinite,data) && minimum(data)>0
                passed=positive && max_scaled_error<0.01
                push!(attempts,(scheme=scheme.name,max_scaled_error=max_scaled_error,relative_l2=relative_l2,passed=passed,seconds=time()-start_time,error=""))
                if passed
                    return (prediction=predicted,attempts=attempts,max_scaled_error=max_scaled_error,relative_l2=relative_l2,scheme=scheme.name)
                end
            catch err
                push!(attempts,(scheme=scheme.name,max_scaled_error=NaN,relative_l2=NaN,passed=false,seconds=time()-start_time,error=sprint(showerror,err)))
            end
        end
        error("PINN не выполнила критерий точности; результаты попыток: $(attempts)")
    end
end

# ╔═╡ 2f349b0d-658f-480f-b66a-c3a5b02414db
begin
    pinn_runs=map(enumerate(competition_cases)) do (i,item)
        map(enumerate(item.starts)) do (j,initial)
            train_competition_checked(item.params,initial,numerical_runs[i][j].reference)
        end
    end
end

# ╔═╡ fdc14c5e-3ece-4810-a06a-c8673cc6b5f9
begin
    training_rows=["| Сценарий / старт | Попытка | Схема | max ошибка / K | L2-ошибка | Критерий | Время, с |", "|:--|--:|:--|--:|--:|:--|--:|"]
    training_changes=String[]
    for (i,item) in enumerate(competition_cases), (j,initial) in enumerate(item.starts)
        result=pinn_runs[i][j]
        for (k,attempt) in enumerate(result.attempts)
            push!(training_rows,@sprintf("| %s / (%.0f, %.0f) | %d | %s | %.6g | %.6g | %s | %.1f |",item.title,initial[1],initial[2],k,attempt.scheme,attempt.max_scaled_error,attempt.relative_l2,attempt.passed ? "пройден" : "не пройден",attempt.seconds))
        end
        if length(result.attempts)>1
            push!(training_changes,"- $(item.title), старт $(initial): первая схема не прошла критерий; применено окно 1 и адаптивное интегрирование, чтобы уменьшить локальную ошибку и проверить невязку между узлами.")
        end
    end
    change_text=isempty(training_changes) ? "Во всех запусках первая схема прошла критерий; повторное обучение не потребовалось." : join(training_changes,"
")
    Markdown.parse("### Фактический журнал обучения

" * join(training_rows,"
") * "

### Какие изменения действительно потребовались
" * change_text)
end

# ╔═╡ e07ff357-2f8e-43ca-8e78-f335df8413c5
md"""
### График 4. PINN и численное решение: совпадают ли траектории
**Что показывает:** сравнение PINN с эталоном Tsit5 на интервале 0–20 во всех пяти запусках.
В заголовке каждой панели указаны сценарий и начальные численности.
**Оси:** время и численность. Синий — вид 1, оранжевый — вид 2;
сплошная линия — Tsit5, пунктир того же цвета — PINN.
**Как читать:** чем ближе пунктир к сплошной линии, тем лучше совпадение.
При малой ошибке линии визуально накладываются, поэтому один этот рисунок не доказывает точность.
Для количественной оценки используйте графики ошибок и увеличенный фрагмент ниже.
Сравнение PINN здесь ограничено временем 20; долгосрочные исходы на графике 1 проверены до времени 60.
"""

# ╔═╡ 1b792fbf-845b-46c4-baa1-d6120926ca4f
begin
    comparison_panels=Plots.Plot[]
    for (i,item) in enumerate(competition_cases), (j,initial) in enumerate(item.starts)
        result=pinn_runs[i][j]
        prediction=result.prediction
        reference=numerical_runs[i][j].reference
        ts=prediction.t
        title=@sprintf("%s; старт (%.0f, %.0f)",item.title,initial[1],initial[2])
        panel=plot(;title=title,xlabel="Время",ylabel="Численность",legend=:right)
        for k in 1:2
            col=k==1 ? :royalblue : :darkorange
            plot!(panel,ts,[reference(t)[k] for t in ts];color=col,linewidth=2,label="Вид $k / Tsit5")
            plot!(panel,ts,prediction[k,:];color=col,linestyle=:dash,linewidth=2,label="Вид $k / PINN")
        end
        push!(comparison_panels,panel)
    end
    push!(comparison_panels,plot(;axis=false,grid=false,legend=false,annotations=(0.5,0.5,text("Сплошная — Tsit5
Пунктир — PINN
Сравнение на 0–20
Пределы проверены на t=60",12))))
    pinn_comparison_figure=plot(comparison_panels...;layout=(3,2),size=(1360,1250),dpi=180,margin=5Plots.mm)
    savefig(pinn_comparison_figure,joinpath(competition_output,"pinn_vs_numeric_all_cases.png"))
    pinn_comparison_figure
end

# ╔═╡ bc391e5a-6fd3-4a34-a9a0-43f63ac26718
md"""
### График 5. Ошибка PINN, нормированная на ёмкость среды
**Что показывает:** величину `|Nᵢ(PINN) − Nᵢ(Tsit5)| / Kᵢ` в каждой точке общей сетки.
**Оси:** по горизонтали — время 0–20, по вертикали — безразмерная ошибка;
синий — вид 1, оранжевый — вид 2.
**Как читать:** ноль означает совпадение в данной точке, пики показывают участки большего расхождения.
Например, `4×10⁻⁵` означает `0,004%` ёмкости среды; при `K=100` это `0,004` особи.
Каждая панель имеет собственный масштаб вертикальной оси — сравнивайте численные деления, а не только высоту пиков.
Это нормировка на постоянную ёмкость среды, а не на текущую численность исчезающего вида.
"""

# ╔═╡ 45f1beff-0cf0-4cfd-ad2b-a5810b73f8a9
begin
    error_panels=Plots.Plot[]
    for (i,item) in enumerate(competition_cases), (j,initial) in enumerate(item.starts)
        prediction=pinn_runs[i][j].prediction
        ref=numerical_runs[i][j].reference
        target=reduce(hcat,[ref(t) for t in prediction.t])
        errors=abs.((Array(prediction)-target)./[item.params.K1,item.params.K2])
        panel=plot(prediction.t,errors[1,:];color=:royalblue,label="Вид 1",title="$(item.title), $(Int.(initial))",xlabel="Время",ylabel="|PINN − Tsit5| / K")
        plot!(panel,prediction.t,errors[2,:];color=:darkorange,label="Вид 2")
        push!(error_panels,panel)
    end
    push!(error_panels,plot(;axis=false,grid=false,legend=false,annotations=(0.5,0.5,text("Нормировка на K
не делит на исчезающую
численность вида",12))))
    pinn_errors_figure=plot(error_panels...;layout=(3,2),size=(1360,1200),dpi=180,margin=5Plots.mm)
    savefig(pinn_errors_figure,joinpath(competition_output,"pinn_errors_all_cases.png"))
    pinn_errors_figure
end

# ╔═╡ 1d65becf-61f6-4a27-bbf9-3c5003c1a1a6
md"""
## Различия, которые не видны на общем графике
Ниже показаны **абсолютные ошибки в единицах численности** для всех пяти запусков
и увеличенный фрагмент с наибольшим точечным расхождением среди всех сценариев и обоих видов.
На увеличении Tsit5 показан синим, PINN — пурпурным пунктиром с маркерами.
Увеличен масштаб оси Y; сами численности показаны в исходных единицах.
"""

# ╔═╡ df2505e7-d839-475d-86a2-d6853ec17b36
md"""
### График 6. Абсолютная ошибка PINN в исходных единицах численности
**Что показывает:** `|Nᵢ(PINN) − Nᵢ(Tsit5)|` для обоих видов во всех пяти запусках.
**Оси:** время 0–20 и абсолютное различие численностей, в особях;
синий — вид 1, оранжевый — вид 2.
**Как читать:** значение `0,003` означает расхождение на `0,003` особи.
Абсолютная величина не показывает, какое решение выше — для этого нужна знаковая разность на графике 7.
Вертикальные масштабы панелей различаются: учитывайте подписи делений.
Это разница двух непрерывных модельных численностей, которые могут принимать дробные значения.
"""

# ╔═╡ 1620be60-5469-4222-b974-3e685cfc6583
begin
    absolute_error_panels=Plots.Plot[]
    zoom_candidates=NamedTuple[]
    for (i,item) in enumerate(competition_cases), (j,initial) in enumerate(item.starts)
        prediction=pinn_runs[i][j].prediction
        reference=numerical_runs[i][j].reference
        target=reduce(hcat,[reference(t) for t in prediction.t])
        delta=Array(prediction)-target
        point=argmax(abs.(delta))
        push!(zoom_candidates,(case_index=i,start_index=j,species=point[1],time_index=point[2],
            time=prediction.t[point[2]],error=abs(delta[point]),signed_error=delta[point]))
        panel=plot(prediction.t,abs.(delta[1,:]);color=:royalblue,label="Вид 1",
            title="$(item.title), старт $(Int.(initial))",xlabel="Время",ylabel="Абсолютная ошибка, особей",
            linewidth=1.4,legend=:outertopright)
        plot!(panel,prediction.t,abs.(delta[2,:]);color=:darkorange,label="Вид 2",linewidth=1.4)
        push!(absolute_error_panels,panel)
    end
    worst_zoom=zoom_candidates[argmax(getproperty.(zoom_candidates,:error))]
    push!(absolute_error_panels,plot(;axis=false,grid=false,legend=false,
        annotations=(0.5,0.5,text(@sprintf("|PINN − Tsit5|
В исходных единицах численности
Максимум по всем запускам:
%.6f особи",worst_zoom.error),12))))
    absolute_error_figure=plot(absolute_error_panels...;layout=(3,2),size=(1360,1200),dpi=180,margin=5Plots.mm)
    savefig(absolute_error_figure,joinpath(competition_output,"pinn_absolute_errors_all_cases.png"))
    absolute_error_figure
end

# ╔═╡ 8b61f133-33e0-496a-984f-187d9f7a37a5
md"""
### График 7. Увеличенный фрагмент с наибольшим расхождением
**Что показывает:** окрестность точки с максимальной абсолютной ошибкой на сохранённой сетке
среди всех сценариев и обоих видов. Сценарий, вид, старт и момент времени указаны после рисунка.
**Верхняя панель:** по горизонтали — время в узком фрагменте, по вертикали — численность
с увеличенным масштабом оси. Синяя линия — Tsit5; пурпурный пунктир с маркерами — PINN.
Обратите внимание: вертикальная ось начинается около равновесной численности, а не с нуля.
**Нижняя панель:** в том же временном фрагменте показана разность `PINN − Tsit5`, в особях.
Серая горизонталь — нулевая разность, чёрная точка — выбранное максимальное абсолютное расхождение.
**Как читать:** выше нулевой линии PINN завышает численность относительно Tsit5, ниже — занижает.
Колебания на увеличении характеризуют небольшую ошибку приближения;
их нельзя принимать за самостоятельные колебания реальных популяций.
"""

# ╔═╡ 7ca5bbf7-07fc-46d8-87ed-0a93d033c842
begin
    zoom_item=competition_cases[worst_zoom.case_index]
    zoom_prediction=pinn_runs[worst_zoom.case_index][worst_zoom.start_index].prediction
    zoom_reference=numerical_runs[worst_zoom.case_index][worst_zoom.start_index].reference
    zoom_species=worst_zoom.species
    zoom_indices=findall(t->abs(t-worst_zoom.time)<=0.4+1e-9,zoom_prediction.t)
    zoom_times=zoom_prediction.t[zoom_indices]
    zoom_dense_times=collect(range(first(zoom_times),last(zoom_times);length=201))
    zoom_numeric=[zoom_reference(t)[zoom_species] for t in zoom_dense_times]
    zoom_neural=zoom_prediction[zoom_species,zoom_indices]
    zoom_delta=[zoom_prediction[zoom_species,index]-zoom_reference(zoom_prediction.t[index])[zoom_species] for index in zoom_indices]
    zoom_all_values=vcat(zoom_numeric,zoom_neural)
    zoom_padding=0.15*max(maximum(zoom_all_values)-minimum(zoom_all_values),worst_zoom.error,1e-6)
    zoom_overlay=plot(zoom_dense_times,zoom_numeric;color=:royalblue,linewidth=2.5,label="Численное решение / Tsit5",
        xlabel="Время",ylabel="Численность, особей",
        title="$(zoom_item.title), вид $zoom_species: увеличенный фрагмент",
        ylims=(minimum(zoom_all_values)-zoom_padding,maximum(zoom_all_values)+zoom_padding),
        yformatter=x->@sprintf("%.5f",x),legend=:outertopright)
    plot!(zoom_overlay,zoom_times,zoom_neural;color=:magenta,linestyle=:dash,linewidth=1.5,
        marker=:circle,markersize=3.5,label="PINN / сохранённые точки")
    zoom_residual=plot(zoom_times,zoom_delta;color=:magenta,linewidth=1.8,marker=:circle,markersize=3,
        xlabel="Время",ylabel="PINN − Tsit5, особей",legend=false,
        title=@sprintf("Разность в том же фрагменте; max |Δ| = %.6f",worst_zoom.error))
    hline!(zoom_residual,[0.0];color=:gray,linestyle=:dash,label="")
    scatter!(zoom_residual,[worst_zoom.time],[worst_zoom.signed_error];color=:black,markersize=5,label="")
    pinn_zoom_figure=plot(zoom_overlay,zoom_residual;layout=(2,1),size=(1050,850),dpi=180,margin=6Plots.mm)
    savefig(pinn_zoom_figure,joinpath(competition_output,"pinn_zoom_largest_difference.png"))
    pinn_zoom_figure
end

# ╔═╡ 495070a3-4dd1-4972-8acb-19c4fcac63e9
md"""
### Какой фрагмент увеличен
**$(zoom_item.title)**, старт **$(competition_cases[worst_zoom.case_index].starts[worst_zoom.start_index])**,
**вид $(worst_zoom.species)**, время **$(round(worst_zoom.time;digits=3))**.
Максимальное абсолютное расхождение на проверочной сетке:
**$(round(worst_zoom.error;sigdigits=7)) особи**.

На верхнем графике различие видно за счёт узкого диапазона оси Y и разных цветов.
На нижнем графике показана знаковая разность: положительное значение означает,
что PINN предсказывает большую численность, чем Tsit5.
Маркеры PINN соответствуют сохранённой сетке с шагом **$(sample_dt)**.
Для этих графиков используются уже рассчитанные траектории.
"""

# ╔═╡ 418422ed-5b2d-49ee-a719-230d0fe63070
begin
    @assert competition_regime.(getproperty.(competition_cases,:params)) == [:species1,:species2,:coexistence,:bistability]
    @assert norm(competition_rhs(interior_equilibrium(competition_cases[3].params),competition_cases[3].params,0.0)) < 1e-10
    @assert maximum(real.(eigvals(competition_jacobian(interior_equilibrium(competition_cases[3].params),competition_cases[3].params)))) < 0
    @assert minimum(real.(eigvals(competition_jacobian(interior_equilibrium(competition_cases[4].params),competition_cases[4].params)))) < 0 < maximum(real.(eigvals(competition_jacobian(interior_equilibrium(competition_cases[4].params),competition_cases[4].params))))
    for group in pinn_runs, result in group
        @assert result.max_scaled_error < 0.01
        @assert minimum(Array(result.prediction)) > 0
        @assert length(result.prediction.t)==401
    end
    competition_summary=(scenarios=4,trajectories=5,
        max_scaled_error=maximum(result.max_scaled_error for group in pinn_runs for result in group),
        max_relative_l2=maximum(result.relative_l2 for group in pinn_runs for result in group),
        retrained_trajectories=count(result->length(result.attempts)>1,[result for group in pinn_runs for result in group]),
        saved_figures=sort(filter(f->endswith(f,".png"),readdir(competition_output))))
end

# ╔═╡ 9876875c-c490-431f-8252-7ba03fd31d64
md"""
## Что установлено и где границы эксперимента
- Для выбранных коэффициентов реализованы все четыре строгих режима.
- Бистабильность показана двумя противоположными исходами при одинаковых коэффициентах.
- Аналитические равновесия проверены численно; эталон Tsit5 сверён с Vern9.
- PINN обучается только по уравнениям и исходным условиям, а точность проверяется отдельно.
- Для положительных ``r_i,K_i`` и выбранной автономной модели не следует ожидать
  циклов хищник–жертва: здесь оба межвидовых воздействия отрицательны.
- Расчёт ``t=60`` — конечная иллюстрация аналитически определённых пределов;
  расчёт PINN проверен только на ``0\le t\le20``.
- Нельзя переносить эти численные коэффициенты на реальные виды без оценки параметров по данным.
- Дегенеративные границы, сезонность, случайность и отдельная динамика запаса ресурса не включены.

Графики сохранены в `results/competition/image` рядом с блокнотом.
"""

# ╔═╡ Cell order:
# ╠═74d537f6-8ffb-4076-ad35-804e55694b86
# ╟─30ddc5af-6ee4-4fa8-9926-b4b63c40f202
# ╠═e723ea08-d14d-4faf-a1f0-23cd8b8e2624
# ╟─e91914bc-e8c0-4464-9452-316d4fcb8c26
# ╠═19e6e84a-05b3-47f4-82a2-e7ce132063f9
# ╠═a063f8a3-7e06-439f-ae7a-2b789de2b1cf
# ╠═f929c6ec-3e95-45e2-888f-c72ea8db8777
# ╠═a448a859-6a30-462b-9a95-d868d33c56b0
# ╟─4157c631-50a9-4c39-96c4-cb422d97731a
# ╠═f0d996eb-e665-4e52-8c7f-9e736e2a29ab
# ╟─a9d55169-689b-40b2-9dd0-81df33992b6f
# ╠═40f6bfe2-6060-4c04-8839-d987371556ac
# ╟─cb74579e-b9e1-41ce-bb18-439e9f977b18
# ╟─9fb23b76-fbdc-47fa-b338-5a29140662c8
# ╠═e9128cf9-4e71-42fa-ac49-244554571682
# ╟─463a104f-12e1-4f64-85ed-a35cf11b1d46
# ╠═54048d1f-1b90-4ec2-897f-b4ee741a1605
# ╠═2f349b0d-658f-480f-b66a-c3a5b02414db
# ╠═fdc14c5e-3ece-4810-a06a-c8673cc6b5f9
# ╟─e07ff357-2f8e-43ca-8e78-f335df8413c5
# ╠═1b792fbf-845b-46c4-baa1-d6120926ca4f
# ╟─bc391e5a-6fd3-4a34-a9a0-43f63ac26718
# ╠═45f1beff-0cf0-4cfd-ad2b-a5810b73f8a9
# ╟─1d65becf-61f6-4a27-bbf9-3c5003c1a1a6
# ╟─df2505e7-d839-475d-86a2-d6853ec17b36
# ╠═1620be60-5469-4222-b974-3e685cfc6583
# ╟─8b61f133-33e0-496a-984f-187d9f7a37a5
# ╠═7ca5bbf7-07fc-46d8-87ed-0a93d033c842
# ╟─495070a3-4dd1-4972-8acb-19c4fcac63e9
# ╠═418422ed-5b2d-49ee-a719-230d0fe63070
# ╟─9876875c-c490-431f-8252-7ba03fd31d64
