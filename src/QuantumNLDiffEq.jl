module QuantumNLDiffEq

import Yao: AbstractBlock, zero_state, expect, dispatch!, dispatch, chain, Add, Scale,
            TimeEvolution, IdentityGate, igate
import Flux: update!, Adam
import Zygote: gradient
import SciMLBase: AbstractODEProblem, AbstractSciMLOperator, ODEFunction
import ForwardDiff: jacobian
import ChainRulesCore: rrule, NoTangent

function ODEEncode(u, p, t, ode::ODEFunction)
    return ode(u, p, t)
end

function rrule(::typeof(ODEEncode), u, p, t, ode::ODEFunction)
    y = ODEEncode(u, p, t, ode)
    function func_pullback(ȳ)
        Ju = jacobian((u)->ODEEncode(u, p, t, ode), u)
        Jp = jacobian((p)->ODEEncode(u, p, t, ode), p)
        NoTangent(), Ju'*ȳ, Jp'*ȳ, NoTangent(), NoTangent()
    end
    return y, func_pullback
end

abstract type AbstractFeatureMap end
abstract type AbstractBoundaryHandling end
abstract type AbstractLoss end
abstract type AbstractRegularisationParams end
abstract type AbstractCostParams end

struct Product <: AbstractFeatureMap end
struct ChebyshevSparse <: AbstractFeatureMap
    pc::Int64
end
struct ChebyshevTower <: AbstractFeatureMap
    pc::Int64
end
Base.@kwdef mutable struct Pinned <: AbstractBoundaryHandling
    eta::Float64 = 1.0
end
struct Floating <: AbstractBoundaryHandling end
struct Optimized <: AbstractBoundaryHandling
    fc::Float64
end
struct NoRegularisation <: AbstractRegularisationParams end
mutable struct RegularisationParams <: AbstractRegularisationParams
    u_reg::Vector{Vector{Float64}}
    M_reg::Vector{Float64}
    reg_param::Float64
end
struct NoCostParams <: AbstractCostParams end
struct CostParams <: AbstractCostParams
    lambda::Vector{Vector{Float64}}
end

Base.@kwdef mutable struct DQCType
    afm::AbstractFeatureMap
    fm::AbstractBlock
    cost::Union{Vector{<:AbstractBlock}, Vector{<:Vector{<:AbstractBlock}}}
    var::AbstractBlock
    N::Int64
    evol::Union{TimeEvolution, IdentityGate} = igate(N)
end

Base.@kwdef mutable struct DQCConfig
    reg::AbstractRegularisationParams = NoRegularisation()
    cost_params::AbstractCostParams = NoCostParams()
    abh::AbstractBoundaryHandling
    loss::Function
end

include("phi.jl")
include("new_circuit.jl")
include("calculate_diff_evalue.jl")
include("calculate_evalue.jl")
include("loss.jl")

function train!(
        DQC::Union{DQCType, Vector{DQCType}}, prob::AbstractODEProblem, config::DQCConfig,
        M::AbstractVector, theta; optimizer = Adam(0.075), steps = 300)
    for _ in 1:steps
        if config.abh isa Optimized
            function conf(fc, config::DQCConfig)
                config.abh = Optimized(fc)
                return config
            end
            fc = config.abh.fc
            grads = gradient((_theta, _fc) -> loss(DQC, prob, conf(_fc, config), M, _theta), theta, fc)
            if DQC isa DQCType
                for (p, g) in zip([theta, [fc]], [grads[1], [grads[2]]])
                    update!(optimizer, p, g)
                end
                dispatch!(DQC.var, theta)
            else
                for (p, g) in zip([theta, [[fc]]], [grads[1], [[grads[2]]]])
                    for (x, y) in zip(p, g)
                        update!(optimizer, x, y)
                    end
                end
                for i in 1:length(DQC)
                    dispatch!(DQC[i].var, theta[i])
                end
            end
            for i in 1:length(DQC)
                dispatch!(DQC[i].var, theta[i])
            end
            config.abh = Optimized(fc)
        else
            grads = gradient(_theta -> loss(DQC, prob, config, M, _theta), theta)[1]
            if DQC isa DQCType
                update!(optimizer, theta, grads)
                dispatch!(DQC.var, theta)
            else
                for (p, g) in zip(theta, grads)
                    update!(optimizer, p, g)
                end
                for i in 1:length(DQC)
                    dispatch!(DQC[i].var, theta[i])
                end
            end
        end
    end
end

function tr_custom!(
        DQC::Union{Vector{DQCType}, DQCType}, prob::AbstractODEProblem, config::DQCConfig,
        M::AbstractVector, theta; optimizer = Adam(0.075), steps = 300)
    for s in 1:steps
        config.reg.reg_param = 1.0 - s/steps
        grads = gradient(_theta -> loss(DQC, prob, config, M, _theta), theta)[1]
        if DQC isa DQCType
            update!(optimizer, theta, grads)
            dispatch!(DQC.var, theta)
        else
            for (p, g) in zip(theta, grads)
                update!(optimizer, p, g)
            end
            for i in 1:length(DQC)
                dispatch!(DQC[i].var, theta[i])
            end
        end
    end
end

export loss, train!, DQCType, DQCConfig
end
