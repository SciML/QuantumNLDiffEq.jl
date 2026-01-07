module QuantumNLDiffEq

import Yao: AbstractBlock, zero_state, expect, dispatch!, dispatch, chain, Add, Scale,
    TimeEvolution, IdentityGate, igate
import Optimisers
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
        Ju = jacobian((u) -> ODEEncode(u, p, t, ode), u)
        Jp = jacobian((p) -> ODEEncode(u, p, t, ode), p)
        return NoTangent(), Ju' * ȳ, Jp' * ȳ, NoTangent(), NoTangent()
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

"""
    DQCType

Differential Quantum Circuit type that encodes the structure of a quantum circuit for solving differential equations.

# Fields
- `afm::AbstractFeatureMap`: Ansatz feature mapping (e.g., `ChebyshevSparse`, `ChebyshevTower`, `Product`)
- `fm::AbstractBlock`: Feature map quantum circuit
- `cost::Union{Vector{<:AbstractBlock}, Vector{<:Vector{<:AbstractBlock}}}`: Cost function observables
- `var::AbstractBlock`: Variational quantum circuit
- `N::Int64`: Number of qubits
- `evol::Union{TimeEvolution, IdentityGate}`: Time evolution operator (default: identity gate)
"""
Base.@kwdef mutable struct DQCType
    afm::AbstractFeatureMap
    fm::AbstractBlock
    cost::Union{Vector{<:AbstractBlock}, Vector{<:Vector{<:AbstractBlock}}}
    var::AbstractBlock
    N::Int64
    evol::Union{TimeEvolution, IdentityGate} = igate(N)
end

"""
    DQCConfig

Configuration for Differential Quantum Circuit training and evaluation.

# Fields
- `reg::AbstractRegularisationParams`: Regularization parameters (default: `NoRegularisation()`)
- `cost_params::AbstractCostParams`: Cost function parameters (default: `NoCostParams()`)
- `abh::AbstractBoundaryHandling`: Boundary handling method (`Floating`, `Pinned`, or `Optimized`)
- `loss::Function`: Loss function for training
"""
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

# Helper to apply gradient updates in-place for our parameter types
function apply_update!(opt_state, theta::Vector{Float64}, grads)
    if grads !== nothing
        new_state, new_theta = Optimisers.update(opt_state, theta, grads)
        theta .= new_theta
        return new_state
    end
    return opt_state
end

function apply_update!(opt_state, theta::Vector{Vector{Float64}}, grads)
    if grads !== nothing
        for i in eachindex(theta)
            if grads[i] !== nothing
                new_state_i,
                    new_theta_i = Optimisers.update(opt_state[i], theta[i], grads[i])
                theta[i] .= new_theta_i
                opt_state[i] = new_state_i
            end
        end
    end
    return opt_state
end

"""
    train!(DQC, prob, config, M, theta; optimizer=Adam(0.075), steps=300)

Train a Differential Quantum Circuit (DQC) to solve an ODE problem.

# Arguments
- `DQC::Union{DQCType, Vector{DQCType}}`: Single DQC or vector of DQCs for multi-equation systems
- `prob::AbstractODEProblem`: ODE problem from DifferentialEquations.jl
- `config::DQCConfig`: Configuration for training (boundary handling, loss function, etc.)
- `M::AbstractVector`: Mesh points for training
- `theta`: Initial parameters for the variational circuit(s)

# Keyword Arguments
- `optimizer=Adam(0.075)`: Flux optimizer for gradient descent
- `steps=300`: Number of training iterations

# Example
```julia
using DifferentialEquations, Yao, QuantumNLDiffEq

prob = ODEProblem((u,p,t) -> -1*p[1]*u*(p[2] + tan(p[1]*t)), [1.0], (0.0, 0.9), [8.0, 0.1])
DQC = [QuantumNLDiffEq.DQCType(
    afm = QuantumNLDiffEq.ChebyshevSparse(2),
    fm = chain(6, [put(i=>Ry(0)) for i in 1:6]),
    cost = [Add([put(6, i=>Z) for i in 1:6])],
    var = dispatch(EasyBuild.variational_circuit(6,5), :random),
    N = 6
)]
config = DQCConfig(abh = QuantumNLDiffEq.Floating(), loss = (a,b) -> (a-b)^2)
M = range(0, stop=0.9, length=20)
params = [Yao.parameters(DQC[1].var)]

QuantumNLDiffEq.train!(DQC, prob, config, M, params)
```
"""
function train!(
        DQC::Union{DQCType, Vector{DQCType}}, prob::AbstractODEProblem, config::DQCConfig,
        M::AbstractVector, theta; optimizer = Optimisers.Adam(0.075), steps = 300
    )
    opt_state = Optimisers.setup(optimizer, theta)

    # For Optimized boundary handling, initialize fc state once outside the loop
    fc = nothing
    fc_state = nothing
    if config.abh isa Optimized
        fc = [config.abh.fc]  # Wrap in array for Optimisers
        fc_state = Optimisers.setup(optimizer, fc)
    end

    for _ in 1:steps
        if config.abh isa Optimized
            function conf(fc_val, config::DQCConfig)
                config.abh = Optimized(fc_val)
                return config
            end
            grads = gradient(
                (
                    _theta, _fc,
                ) -> loss(DQC, prob, conf(_fc[1], config), M, _theta), theta, fc
            )
            opt_state = apply_update!(opt_state, theta, grads[1])
            if grads[2] !== nothing
                fc_state, new_fc = Optimisers.update(fc_state, fc, grads[2])
                fc[1] = new_fc[1]
            end
            if DQC isa DQCType
                dispatch!(DQC.var, theta)
            else
                for i in 1:length(DQC)
                    dispatch!(DQC[i].var, theta[i])
                end
            end
            config.abh = Optimized(fc[1])
        else
            grads = gradient(_theta -> loss(DQC, prob, config, M, _theta), theta)[1]
            opt_state = apply_update!(opt_state, theta, grads)
            if DQC isa DQCType
                dispatch!(DQC.var, theta)
            else
                for i in 1:length(DQC)
                    dispatch!(DQC[i].var, theta[i])
                end
            end
        end
    end
    return
end

function tr_custom!(
        DQC::Union{Vector{DQCType}, DQCType}, prob::AbstractODEProblem, config::DQCConfig,
        M::AbstractVector, theta; optimizer = Optimisers.Adam(0.075), steps = 300
    )
    opt_state = Optimisers.setup(optimizer, theta)
    for s in 1:steps
        config.reg.reg_param = 1.0 - s / steps
        grads = gradient(_theta -> loss(DQC, prob, config, M, _theta), theta)[1]
        opt_state = apply_update!(opt_state, theta, grads)
        if DQC isa DQCType
            dispatch!(DQC.var, theta)
        else
            for i in 1:length(DQC)
                dispatch!(DQC[i].var, theta[i])
            end
        end
    end
    return
end

export loss, train!, DQCType, DQCConfig
end
