# Классическая модель межвидовой конкуренции Лотки-Вольтерры
# dN1/dt = r1*N1*(1 - (N1 + α12*N2)/K1)
# dN2/dt = r2*N2*(1 - (N2 + α21*N1)/K2)

using OrdinaryDiffEq, Plots

function competition!(du, u, p, t)
    N1, N2 = u
    r1, r2, K1, K2, α12, α21 = p

    du[1] = r1 * N1 * (1 - (N1 + α12 * N2) / K1)
    du[2] = r2 * N2 * (1 - (N2 + α21 * N1) / K2)
end

# --- Режим 1: Устойчивое сосуществование ---
# α12*α21 < 1: межвидовая конкуренция слабее внутривидовой
p_coexist = [1.0, 1.0, 100.0, 100.0, 0.5, 0.3]
u0 = [10.0, 10.0]
tspan = (0.0, 50.0)

prob1 = ODEProblem(competition!, u0, tspan, p_coexist)
sol1 = solve(prob1, Tsit5(), saveat=0.1)

# --- Режим 2: Вытеснение вида 2 видом 1 ---
p_exclude = [1.0, 1.0, 100.0, 80.0, 0.4, 1.5]

prob2 = ODEProblem(competition!, u0, tspan, p_exclude)
sol2 = solve(prob2, Tsit5(), saveat=0.1)

# --- Режим 3: Бистабильность ---
# α12*α21 > 1: результат зависит от начальных условий
p_bistable = [1.0, 1.0, 100.0, 100.0, 1.5, 1.5]

prob3a = ODEProblem(competition!, [80.0, 20.0], tspan, p_bistable)
sol3a = solve(prob3a, Tsit5(), saveat=0.1)

prob3b = ODEProblem(competition!, [20.0, 80.0], tspan, p_bistable)
sol3b = solve(prob3b, Tsit5(), saveat=0.1)
