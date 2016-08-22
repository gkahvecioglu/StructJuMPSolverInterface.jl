push!(LOAD_PATH,"/nfs2/gkahvecioglu/StructJuMP/src")	# path for StructJuMP 
push!(LOAD_PATH,"/homes/gkahvecioglu/StructJuMPSolverInterface.jl/src") #path for StructJuMPSolverInterface
using StructJuMP, JuMP

firststage = StructuredModel()
@variable(firststage, x[1:2])
@constraint(firststage, sum(x) == 100)
@NLobjective(firststage, Min, x[1] + x[2])

for scen in 1:1
    #bl = StructuredModel(parent=firststage)
    @variable(firststage, y[1:2])
    @constraint(firststage, x[3-scen] + sum(y) ≥  0)
    @constraint(firststage, x[3-scen] + sum(y) ≤ 50)
    @NLobjective(firststage, Min, y[1] + y[2])
end

solve(firststage, solver="Minotaur")
