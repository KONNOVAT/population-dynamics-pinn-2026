import Pkg
Pkg.activate(joinpath(@__DIR__, "pluto_env"))
using Pluto
Pluto.run(notebook=joinpath(@__DIR__, "lv_pinn.jl"), host="127.0.0.1",
    port=1234, launch_browser=false)
