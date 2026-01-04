function calc_cost(DQC::DQCType, ::NoCostParams)
    return DQC.cost
end

function calc_cost(DQC::Vector{DQCType}, ::NoCostParams)
    return [DQC[i].cost for i in 1:length(DQC)]
end

function calc_cost(DQC::DQCType, cost_params::CostParams)
    no_eqns = length(cost_params.lambda)
    length_range = length.(cost_params.lambda)

    return [
        [Scale(DQC.cost[i][j], cost_params.lambda[i][j]) for j in 1:length_range[i]]
            for i in 1:no_eqns
    ]
end

function calc_cost(DQC::Vector{DQCType}, cost_params::CostParams)
    no_eqns = length(DQC)
    length_range = length.(cost_params.lambda)

    return [[Scale(DQC[i].cost[j]) for j in 1:length_range[i]] for i in 1:no_eqns]
end

function loss_diff(
        DQC::Vector{DQCType}, prob::AbstractODEProblem, M::AbstractVector,
        cost_params::AbstractCostParams, abh::AbstractBoundaryHandling,
        loss::Function, theta::Vector{Vector{Float64}}
    )
    no_eqns = length(DQC)
    if no_eqns != length(prob.u0)
        throw("Number of encoded equations don't match number of DQCs")
    end
    cost = calc_cost(DQC, cost_params)
    Loss_diff = ComplexF64(0.0)
    for x in 1:length(M)
        evalue = [
            calculate_evalue(DQC[i], cost[i], prob.u0[i], abh, theta[i], M[x], M[1])
                for i in 1:no_eqns
        ]

        diff_evalue = [
            calculate_diff_evalue(DQC[i], cost[i], theta[i], M[x])
                for i in 1:no_eqns
        ]

        du = ODEEncode(real.(evalue), prob.p, M[x], prob.f)
        loss_ind = [loss(diff_evalue[i], du[i]) for i in 1:no_eqns]

        Loss_diff += sum(loss_ind)
    end
    return real(Loss_diff / length(M))
end

function loss_diff(
        DQC::DQCType, prob::AbstractODEProblem, M::AbstractVector,
        cost_params::AbstractCostParams, abh::AbstractBoundaryHandling,
        loss::Function, theta::Vector{Float64}
    )
    no_eqns = length(prob.u0)
    if !(DQC.cost isa Vector{<:Vector{<:AbstractBlock}})
        throw("cost functions not defined properly")
    elseif no_eqns != length(DQC.cost)
        throw("Number of encoded equations don't match number of cost functions")
    end
    Loss_diff = ComplexF64(0.0)
    cost = calc_cost(DQC, cost_params)
    for x in 1:length(M)
        evalue = [
            calculate_evalue(DQC, cost[i], prob.u0[i], abh, theta, M[x], M[1])
                for i in 1:no_eqns
        ]

        diff_evalue = [calculate_diff_evalue(DQC, cost[i], theta, M[x]) for i in 1:no_eqns]

        du = ODEEncode(real.(evalue), prob.p, M[x], prob.f)
        loss_ind = [loss(diff_evalue[i], du[i]) for i in 1:no_eqns]

        Loss_diff += sum(loss_ind)
    end
    return real(Loss_diff / length(M))
end

function loss_reg(
        ::Union{DQCType, Vector{DQCType}}, ::Vector{Float64}, ::NoRegularisation,
        ::AbstractCostParams, ::AbstractBoundaryHandling, ::Function, ::Any
    )
    return 0.0
end

function loss_reg(
        DQC::DQCType, u0::Vector{Float64}, rp::RegularisationParams,
        cost_params::AbstractCostParams, abh::AbstractBoundaryHandling,
        loss::Function, theta::Vector{Float64}
    )
    no_eqns = length(u0)
    cost = calc_cost(DQC, cost_params)
    Loss_reg = ComplexF64(0.0)
    for x in 1:length(rp.M_reg)
        evalue = [
            calculate_evalue(
                    DQC, cost[i], u0[i], abh, theta, rp.M_reg[x], rp.M_reg[1]
                )
                for i in 1:no_eqns
        ]
        loss_ind = [loss(evalue[i], rp.u_reg[i][x]) for i in 1:no_eqns]
        Loss_reg += sum(loss_ind)
    end
    return real(Loss_reg)
end

function loss_reg(
        DQC::Vector{DQCType}, u0::Vector{Float64}, rp::RegularisationParams,
        cost_params::AbstractCostParams, abh::AbstractBoundaryHandling,
        loss::Function, theta::Vector{Vector{Float64}}
    )
    no_eqns = length(DQC)
    cost = calc_cost(DQC, cost_params)
    Loss_reg = ComplexF64(0.0)
    for x in 1:length(rp.M_reg)
        evalue = [
            calculate_evalue(
                    DQC[i], cost[i], u0[i], abh, theta[i], rp.M_reg[x], rp.M_reg[1]
                )
                for i in 1:no_eqns
        ]
        loss_ind = [loss(evalue[i], rp.u_reg[i][x]) for i in 1:no_eqns]
        Loss_reg += sum(loss_ind)
    end
    return real(Loss_reg)
end

function loss_bound(
        ::Union{DQCType, Vector{DQCType}}, ::Real, ::Vector{Float64},
        ::AbstractCostParams, ::Union{Floating, Optimized}, ::Function, ::Any
    )
    return 0.0
end

function loss_bound(
        DQC::DQCType, M::Real, u0::Vector{Float64}, cost_params::AbstractCostParams,
        abh::Pinned, loss::Function, theta::Vector{Float64}
    )
    no_eqns = length(u0)
    cost = calc_cost(DQC, cost_params)
    evalue = [calculate_evalue(DQC, cost[i], u0[i], abh, theta, M, M) for i in 1:no_eqns]
    loss_ind = abh.eta * sum([loss(evalue[i], u0[i]) for i in 1:no_eqns])
    return real(loss_ind)
end

function loss_bound(
        DQC::Vector{DQCType}, M::Real, u0::Vector{Float64}, cost_params::AbstractCostParams,
        abh::Pinned, loss::Function, theta::Vector{Vector{Float64}}
    )
    no_eqns = length(u0)
    cost = calc_cost(DQC, cost_params)
    evalue = [
        calculate_evalue(DQC[i], cost[i], u0[i], abh, theta[i], M, M)
            for i in 1:no_eqns
    ]
    loss_ind = abh.eta * sum([loss(evalue[i], u0[i]) for i in 1:no_eqns])
    return real(loss_ind)
end

function loss(
        DQC::Union{DQCType, Vector{DQCType}}, prob::AbstractODEProblem,
        config::DQCConfig, M::AbstractVector, theta
    )
    return loss_diff(DQC, prob, M, config.cost_params, config.abh, config.loss, theta) +
        loss_reg(
        DQC, prob.u0, config.reg, config.cost_params,
        config.abh, config.loss, theta
    ) +
        loss_bound(
        DQC, M[1], prob.u0, config.cost_params, config.abh, config.loss, theta
    )
end
