#
# Interface for Minotaur serial
#
include("minotaur_serial_cfunc.jl")

module  MinotaurSolverInterfaceSerial

using MinotaurSolverSerial

using StructJuMP, JuMP
using StructJuMPSolverInterface

import MathProgBase

type NonStructJuMPModel <: ModelInterface
    model::JuMP.Model 
    jac_I::Vector{Int}
    jac_J::Vector{Int}
    hess_I::Vector{Int}
    hess_J::Vector{Int}
    nz_jac::Vector{Int}
    nz_hess::Vector{Int}
    nzj::Int
    nzh::Int
    g_iter::Int

    write_solution::Function
    get_x0::Function
    numvars::Function
    numcons::Function
    nele_jac::Function
    nele_hess::Function
    get_bounds::Function
    eval_f::Function
    eval_g::Function
    eval_grad_f::Function
    eval_jac_g::Function
    eval_h::Function

    function NonStructJuMPModel(model)
        instance = new(model, 
            Vector{Int}(), Vector{Int}(), Vector{Int}(), Vector{Int}(),
            Vector{Int}(), Vector{Int}(), 0 , 0, 0
            )
        
        instance.write_solution = function(x)
            @assert length(x) == getTotalNumVars(m)
            m = instance.model
            idx = 1
            for i = 0:num_scenarios(m)
                mm = getModel(m,i)
                for j = 1:getNumVars(m,i)
                    setvalue(Variable(mm,j),x[idx])
                    idx += 1
                end
            end
        end

        instance.get_x0 = function(x)
            @assert length(x) == getTotalNumVars(m)
            m = instance.model
            idx = 1
            for i = 0:num_scenarios(m)
                mm = getModel(m,i)
                for j = 1:getNumVars(m,i)
                    v_j = getvalue(Variable(mm,j))
                    x[idx] = isnan(v_j)? 1.0:v_j
                    idx += 1
                end
            end
            # @show x
            return x
        end
        instance.numvars = function()
            return getTotalNumVars(instance.model)
        end

        instance.numcons = function()
            return getTotalNumCons(instance.model)
        end

        instance.nele_jac = function()
            @assert length(instance.jac_I) == length(instance.jac_J)
            mat = sparse(instance.jac_I,instance.jac_J,ones(Float64,length(instance.jac_I)))
            instance.nzj = length(mat.nzval)
            return instance.nzj
        end

        instance.nele_hess = function()
            @assert length(instance.hess_I) == length(instance.hess_J)
            mat = sparse(instance.hess_J,instance.hess_I,ones(Float64,length(instance.hess_I)))
            instance.nzh = length(mat.nzval)
            return instance.nzh
        end

        instance.get_bounds = function()
            m = instance.model
            nvar = getTotalNumVars(m)
            ncon = getTotalNumCons(m)
            x_L = Vector{Float64}(nvar)
            x_U = Vector{Float64}(nvar)
            g_L = Vector{Float64}(ncon)
            g_U = Vector{Float64}(ncon)

            row_start = 1
            col_start = 1
            for i = 0:num_scenarios(m)
                mm = getModel(m,i)
                nx = getNumVars(m,i)
                array_copy(mm.colUpper, 1, x_U, col_start, nx)
                array_copy(mm.colLower, 1, x_L, col_start, nx)

                lb,ub = JuMP.constraintbounds(mm)
                ncons = getNumCons(m,i)
                array_copy(lb,1,g_L,row_start,ncons)
                array_copy(ub,1,g_U,row_start,ncons)

                row_start += getNumCons(m,i)
                col_start += getNumVars(m,i)
            end
            # @show g_L, g_U
            return x_L, x_U, g_L, g_U
        end
        
        instance.eval_f = function(x)
            m = instance.model
            obj = 0.0
            start_idx = 1
            for i=0:num_scenarios(m)
                x_new = strip_x(m,i,x,start_idx)    
                obj += MathProgBase.eval_f(get_nlp_evaluator(m,i),x_new)
                start_idx += getNumVars(m,i)
            end
            # @printf("++++++++++++++++ eval_f  \n")
            # @show obj
            # @show x
            return obj;
        end 

        instance.eval_g = function(x,g)
            m = instance.model
            @assert length(g) == getTotalNumCons(m)
            start_idx = 1
            g_start_idx = 1
            for i=0:num_scenarios(m)
                x_new = strip_x(instance.model,i,x,start_idx)
                ncon = getNumCons(m,i)    
                g_new = Vector{Float64}(ncon)
                e = get_nlp_evaluator(m,i)
                MathProgBase.eval_g(e,g_new,strip_x(instance.model,i,x,start_idx))
                array_copy(g_new,1,g,g_start_idx,ncon)
                g_start_idx += ncon
                start_idx += getNumVars(m,i)
            end
            # @printf("+++++++++++++ eval_g \n")
            # @show x , g
        end

        instance.eval_grad_f = function(x,grad_f)
            fill!(grad_f,0.0)
            m = instance.model
            start_idx = 1
            for i=0:num_scenarios(m)
                x_new = strip_x(instance.model,i,x,start_idx)
                e = get_nlp_evaluator(m,i)

                g_f = Vector{Float64}(length(x_new))
                MathProgBase.eval_grad_f(e,g_f,x_new)
                nx = getNumVars(m,i) 

                array_copy(g_f,1,grad_f,start_idx,nx)

                othermap = getStructure(getModel(m,i)).othermap
                for i in othermap
                    pid = i[1].col
                    cid = i[2].col
                    grad_f[pid] += g_f[cid]
                    @assert pid <= getNumVars(m,0)
                end
                start_idx += nx
            end
            # @printf("+++++++++++ eval_grad_f \n")
            # @show  x, grad_f
        end

        instance.eval_jac_g = function(x,mode,rows,cols,nzvals) #x, mode, irows, kcols, values)
            # @printf("+++++++++++ eval_jac_g %s \n", mode) 
            m = instance.model
            if mode==:Structure
                mat = sparse(instance.jac_I,instance.jac_J,ones(Float64,length(instance.jac_I)))
                @assert length(mat.nzval) == instance.nzj
                array_copy(mat.rowval,1,rows,1,length(mat.rowval))
                array_copy(mat.colptr,1,cols,1,length(mat.colptr))
            else
                start_idx = 1
                value_start = 1
                value = Vector{Float64}(length(instance.jac_I))
                for i = 0:num_scenarios(m)
                    e = get_nlp_evaluator(m,i)
                    x_new = strip_x(instance.model,i,x,start_idx)
                    i_nz_jac = instance.nz_jac[i+1]
                    jac_g = Vector{Float64}(i_nz_jac)
                    MathProgBase.eval_jac_g(e,jac_g,x_new)
                    array_copy(jac_g,1,value,value_start,i_nz_jac)
                    nx = getNumVars(m,i)
                    start_idx += nx
                    value_start += i_nz_jac
                end
                @assert length(instance.jac_I) == length(instance.jac_J) == length(value)
                mat = sparse(instance.jac_I,instance.jac_J,value, getTotalNumCons(instance.model), getTotalNumVars(instance.model), keepzeros=true)
                
                jac_I = instance.jac_I
                jac_J = instance.jac_J
                # @printf( "m=%d; n=%d; \n", getTotalNumCons(instance.model), getTotalNumVars(instance.model))
                # @show jac_I, jac_J, value
                # @printf( "sjac=sparse(jac_I,jac_J,value,m,n); \n")

                @assert length(mat.nzval) == instance.nzj
                array_copy(mat.nzval,1,nzvals,1,instance.nzj)
            end
        end

        instance.eval_h = function(x, mode, rows, cols, obj_factor, lambda, nzvals) #x, mode, irows, kcols, obj_factor, lambda, values)
            # @printf("+++++++++++ eval_h - %s, %f \n", mode, obj_factor)
            # @show x
            # @show lambda
            m = instance.model
            if mode == :Structure
                mat = sparse(instance.hess_J,instance.hess_I,ones(Float64,length(instance.hess_I)))
                # @show mat
                @assert length(mat.nzval) == instance.nzh
                array_copy(mat.rowval,1,rows,1,length(mat.rowval))
                array_copy(mat.colptr,1,cols,1,length(mat.colptr))
            else
                start_idx = 1
                value_start = 1
                lambda_start = 1
                value = Vector{Float64}(length(instance.hess_I))
                for i = 0:num_scenarios(m)
                    e = get_nlp_evaluator(m,i)
                    x_new = strip_x(instance.model,i,x,start_idx)
                    nc = getNumCons(m,i)
                    lambda_new = Vector{Float64}(nc)
                    array_copy(lambda,lambda_start, lambda_new, 1, nc)
                    i_nz_hess = instance.nz_hess[i+1]
                    h = Vector{Float64}(i_nz_hess)
                    # @show x
                    # @show x_new
                    # @show obj_factor
                    # @show lambda_new
                    MathProgBase.eval_hesslag(e,h,x_new,obj_factor,lambda_new)
                    # @show h
                    array_copy(h,1,value,value_start,i_nz_hess)
                    nx = getNumVars(m,i)
                    start_idx += nx
                    lambda_start += nc
                    value_start += i_nz_hess
                end
                # @show value
                @assert length(instance.hess_I) == length(instance.hess_J) == length(value)
                mat = sparse(instance.hess_J,instance.hess_I,value,  getTotalNumVars(instance.model), getTotalNumVars(instance.model), keepzeros=true)
                @assert length(mat.nzval) == instance.nzh
                array_copy(mat.nzval,1,nzvals,1,instance.nzh)

                hess_I = instance.hess_J
                hess_J = instance.hess_I
                # @printf("m=%d; n=%d; \n", getTotalNumVars(instance.model), getTotalNumVars(instance.model))
                # @show hess_I, hess_J, value
                # @printf("shess=sparse(hess_I,hess_J,value,m,n); \n")

                # write_x("pips",instance.g_iter,x)
                instance.g_iter += 1
                # @show nzvals
            end
        end

        # initialization  jac
        col_offset = 0
        row_offset = 0
        m = instance.model
        for i = 0:num_scenarios(m)
            reverse_map = Dict{Int,Int}()
            mm = getModel(m,i)
            for ety in getStructure(mm).othermap
                reverse_map[ety[2].col] = ety[1].col #child->parent
            end
            # @show reverse_map
            e = get_nlp_evaluator(m,i)
            # @show "after e"
            i_jac_I, i_jac_J =  MathProgBase.jac_structure(e)
            # @show "after strct jac"
            for idx = 1:length(i_jac_J)
                jj = i_jac_J[idx]
                if haskey(reverse_map,jj)
                    push!(instance.jac_J, reverse_map[jj])
                else
                    push!(instance.jac_J, jj + col_offset)
                end
                push!(instance.jac_I, i_jac_I[idx] + row_offset)
            end
            push!(instance.nz_jac, length(i_jac_J)) #offset by 1

            col_offset += getNumVars(m,i)
            row_offset += getNumCons(m,i)
        end

        #initialization hess
        offset = 0
        for i = 0:num_scenarios(m)
            reverse_map = Dict{Int,Int}()
            mm = getModel(m,i)
            for ety in getStructure(mm).othermap
                reverse_map[ety[2].col] = ety[1].col #child->parent
            end

            e = get_nlp_evaluator(m,i)
            i_hess_I, i_hess_J =  MathProgBase.hesslag_structure(e)
            for idx = 1:length(i_hess_I)
                ii = i_hess_I[idx]
                jj = i_hess_J[idx]

                if haskey(reverse_map,ii)
                    new_ii = reverse_map[ii]
                else
                    new_ii = ii + offset
                end

                if haskey(reverse_map,jj)
                    new_jj = reverse_map[jj]
                else
                    new_jj = jj + offset
                end

                if(new_ii>new_jj)
                    push!(instance.hess_I,new_ii)
                    push!(instance.hess_J,new_jj)
                else
                    push!(instance.hess_I,new_jj)
                    push!(instance.hess_J,new_ii)
                end
            end
            push!(instance.nz_hess, length(i_hess_I)) #offset by 1

            offset += getNumVars(m,i)
        end
        return instance  
    end
end


#######
# Linking with Minotaur Julia Structure interface
######

function structJuMPSolve(model; suppress_warmings=false,kwargs...)
    # @show typeof(model)
    nm = NonStructJuMPModel(model)
    x_L, x_U, g_L, g_U = nm.get_bounds()
    n = getTotalNumVars(model)
    m = getTotalNumCons(model)
    nele_jac = nm.nele_jac()
    nele_hess = nm.nele_hess()

    # @show x_L, x_U
    # @show g_L, g_U
    # @show n,m
    # @show nele_jac,nele_hess

    prob = MinotaurSolverSerial.createProblem(n, m, x_L, x_U, g_L, g_U, nele_jac, nele_hess,
                         nm.eval_f, nm.eval_g, nm.eval_grad_f, nm.eval_jac_g, nm.eval_h)
    # setProblemScaling(prob,1.0)
    nm.get_x0(prob.x)
    status = MinotaurSolverSerial.solveProblem(prob)
    nm.write_solution(prob.x)
    
    return PIPSRetCodeToSolverInterfaceCode[status]
end


KnownSolvers["Minotaur"] = MinotaurSolverInterfaceSerial.structJuMPSolve

end

