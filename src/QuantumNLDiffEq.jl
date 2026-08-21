module QuantumNLDiffEq

import Yao: AbstractBlock, zero_state, expect, dispatch!, dispatch, chain, Add, Scale,
    TimeEvolution, IdentityGate, igate, EasyBuild, Ry, Z, nparameters, parameters, put
import Optimisers
import Zygote: gradient, pullback
import SciMLBase: AbstractODEProblem, ODEFunction, ODEProblem
import ChainRulesCore: rrule, NoTangent
import SciMLPublic: @public

function ODEEncode(u, p, t, ode::ODEFunction)
    return ode(u, p, t)
end

function rrule(::typeof(ODEEncode), u, p, t, ode::ODEFunction)
    y = ODEEncode(u, p, t, ode)
    function func_pullback(ȳ)
        _, pullback_u = pullback((_u) -> ode(_u, p, t), u)
        _, pullback_p = pullback((_p) -> ode(u, _p, t), p)
        return NoTangent(), pullback_u(ȳ)[1], pullback_p(ȳ)[1], NoTangent(), NoTangent()
    end
    return y, func_pullback
end

"""
    AbstractFeatureMap

Developer extension interface for mappings from a scalar mesh coordinate to
the angles of a Differential Quantum Circuit.

# Rules

Packages that own a feature-map subtype may extend [`phi`](@ref), [`load`](@ref),
and [`map_to_circuit`](@ref) for that subtype. `phi(x, map)` must return a
scalar angle for every mesh value accepted by the application. `load(x, N,
map)` must return exactly `N` angles, and `map_to_circuit(a, j, map)` must
return the factor that converts the derivative of `phi` into the contribution
for circuit index `j`. Extensions must only add methods whose feature-map type
they own.
"""
abstract type AbstractFeatureMap end

"""
    AbstractBoundaryHandling

Developer extension interface for the boundary term used by [`loss`](@ref).

# Rules

Packages that own a subtype may extend [`calculate_evalue`](@ref) and
[`loss_bound`](@ref) for it. `calculate_evalue` must return the real-valued
state prediction at the requested mesh point. `loss_bound` must return a
nonnegative real penalty and must not mutate the circuit, problem, mesh, or
parameter inputs. Extensions must only add methods for boundary types they
own.
"""
abstract type AbstractBoundaryHandling end

"""
    AbstractLoss

Internal marker reserved for future loss-object support. It is not a supported
extension point; pass a callable through the `loss` field of [`DQCConfig`](@ref)
instead.
"""
abstract type AbstractLoss end

"""
    AbstractRegularisationParams

Internal supertype for the regularization configurations accepted by
[`DQCConfig`](@ref). Use [`NoRegularisation`](@ref) or
[`RegularisationParams`](@ref); custom subtypes are not a supported extension
point.
"""
abstract type AbstractRegularisationParams end

"""
    AbstractCostParams

Internal supertype for cost-observable weighting configurations accepted by
[`DQCConfig`](@ref). Use [`NoCostParams`](@ref) or [`CostParams`](@ref); custom
subtypes are not a supported extension point.
"""
abstract type AbstractCostParams end

"""
    Product()

Feature map that encodes each mesh coordinate as `asin(x)` on every qubit.

# Examples

```julia
using QuantumNLDiffEq

map = QuantumNLDiffEq.Product()
```
"""
struct Product <: AbstractFeatureMap end

"""
    ChebyshevSparse(pc)

Feature map that encodes `pc * acos(x)` identically on every qubit.

# Arguments

- `pc`: Integer Chebyshev order used to scale the encoded angle.

# Fields

- `pc`: Chebyshev order.

# Examples

```julia
using QuantumNLDiffEq

map = QuantumNLDiffEq.ChebyshevSparse(2)
```
"""
struct ChebyshevSparse <: AbstractFeatureMap
    pc::Int64
end

"""
    ChebyshevTower(pc)

Feature map that encodes the `i`th qubit with `i * pc * acos(x)`.

# Arguments

- `pc`: Integer Chebyshev order used to scale the encoded angle.

# Fields

- `pc`: Chebyshev order.

# Examples

```julia
using QuantumNLDiffEq

map = QuantumNLDiffEq.ChebyshevTower(2)
```
"""
struct ChebyshevTower <: AbstractFeatureMap
    pc::Int64
end

"""
    Pinned(eta = 1.0)

Boundary-handling strategy that penalizes the difference between the circuit
prediction and the initial value at the first mesh point.

# Keyword Arguments

- `eta`: Nonnegative multiplier for the boundary penalty. Defaults to `1.0`.

# Fields

- `eta`: Boundary-penalty multiplier.

# Examples

```julia
using QuantumNLDiffEq

boundary = QuantumNLDiffEq.Pinned(2.0)
```
"""
Base.@kwdef mutable struct Pinned <: AbstractBoundaryHandling
    eta::Float64 = 1.0
end

"""
    Floating()

Boundary-handling strategy that offsets circuit predictions by the initial
state, imposing no separate boundary penalty.

# Examples

```julia
using QuantumNLDiffEq

boundary = QuantumNLDiffEq.Floating()
```
"""
struct Floating <: AbstractBoundaryHandling end

"""
    Optimized(fc)

Boundary-handling strategy with a trainable scalar offset.

# Arguments

- `fc`: Initial scalar offset. [`train!`](@ref) updates this value while
  optimizing the circuit parameters.

# Fields

- `fc`: Current scalar offset.

# Examples

```julia
using QuantumNLDiffEq

boundary = QuantumNLDiffEq.Optimized(0.0)
```
"""
struct Optimized <: AbstractBoundaryHandling
    fc::Float64
end

"""
    NoRegularisation()

Configuration that disables regularization against known solution values.

# Examples

```julia
using QuantumNLDiffEq

reg = QuantumNLDiffEq.NoRegularisation()
```
"""
struct NoRegularisation <: AbstractRegularisationParams end

"""
    RegularisationParams(u_reg, M_reg, reg_param)

Regularization data that penalizes disagreement with known state values.

# Arguments

- `u_reg`: Known state values grouped by encoded equation and mesh point.
- `M_reg`: Mesh points corresponding to each value in `u_reg`.
- `reg_param`: Scalar regularization weight used by custom training schedules.

# Fields

- `u_reg`: Known state values grouped by equation.
- `M_reg`: Regularization mesh points.
- `reg_param`: Mutable regularization weight.

# Examples

```julia
using QuantumNLDiffEq

reg = QuantumNLDiffEq.RegularisationParams([[1.0, 0.5]], [0.0, 1.0], 0.1)
```
"""
mutable struct RegularisationParams <: AbstractRegularisationParams
    u_reg::Vector{Vector{Float64}}
    M_reg::Vector{Float64}
    reg_param::Float64
end

"""
    NoCostParams()

Configuration that uses the cost observables stored in [`DQCType`](@ref)
without additional weights.

# Examples

```julia
using QuantumNLDiffEq

weights = QuantumNLDiffEq.NoCostParams()
```
"""
struct NoCostParams <: AbstractCostParams end

"""
    CostParams(lambda)

Weights for the cost observables used by [`loss`](@ref).

# Arguments

- `lambda`: Observable weights grouped by encoded equation.

# Fields

- `lambda`: Observable weights grouped by encoded equation.

# Examples

```julia
using QuantumNLDiffEq

weights = QuantumNLDiffEq.CostParams([[1.0, 0.5]])
```
"""
struct CostParams <: AbstractCostParams
    lambda::Vector{Vector{Float64}}
end

"""
    DQCType(; afm, fm, cost, var, N, evol = igate(N))

Describe a Differential Quantum Circuit (DQC) used by [`loss`](@ref) and
[`train!`](@ref).

Use one `DQCType` for a scalar differential equation, or a vector of
`DQCType`s for a system where each state component is represented by its own
circuit.

# Fields

- `afm`: Ansatz feature map, such as `QuantumNLDiffEq.ChebyshevSparse`.
- `fm`: Feature-map quantum circuit.
- `cost`: Cost observable or observables. For a single circuit encoding
  multiple equations, pass a vector of observable vectors.
- `var`: Variational quantum circuit.
- `N`: Number of qubits.
- `evol`: Optional time-evolution block. Defaults to the identity gate on `N`
  qubits.

# Examples

```julia
using QuantumNLDiffEq, Yao

DQC = DQCType(
    afm = QuantumNLDiffEq.ChebyshevSparse(2),
    fm = chain(6, [put(i => Ry(0)) for i in 1:6]),
    cost = [Add([put(6, i => Z) for i in 1:6])],
    var = dispatch(EasyBuild.variational_circuit(6, 5), :random),
    N = 6,
)
```
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
    DQCConfig(; reg = NoRegularisation(), cost_params = NoCostParams(), abh, loss)

Configure the loss terms and boundary handling used by [`loss`](@ref) and
[`train!`](@ref).

`DQCConfig` keeps the training objective separate from the circuit definition
in [`DQCType`](@ref). The `loss` field should be a scalar penalty function that
compares a predicted value with a target value.

# Fields

- `reg`: Regularization data for matching known solution values. Defaults to no
  regularization.
- `cost_params`: Weights for cost observables. Defaults to unweighted costs.
- `abh`: Boundary handling strategy, such as `QuantumNLDiffEq.Floating()`.
- `loss`: Function called as `loss(predicted, target)` for each penalty term.

# Examples

```julia
using QuantumNLDiffEq

config = DQCConfig(
    abh = QuantumNLDiffEq.Floating(),
    loss = (predicted, target) -> abs2(predicted - target),
)
```
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

# User-facing configuration types remain qualified rather than exported so a
# circuit setup makes its QuantumNLDiffEq dependency explicit.
@public Product, ChebyshevSparse, ChebyshevTower, Pinned, Floating, Optimized,
    NoRegularisation, RegularisationParams, NoCostParams, CostParams

# Versioned hooks for feature-map and boundary implementations. These are
# intended for packages that own such extensions, not ordinary applications.
@public AbstractFeatureMap, AbstractBoundaryHandling, phi, load, map_to_circuit,
    calculate_evalue, loss_bound

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
    train!(DQC, prob, config, M, theta; optimizer = Optimisers.Adam(0.075), steps = 300)

Train one or more Differential Quantum Circuits against an ODE problem.

`train!` minimizes [`loss`](@ref) over the mesh points `M`, updates `theta` in
place, and dispatches the updated parameters back into the variational circuit.
It returns `nothing`.

# Arguments

- `DQC`: A [`DQCType`](@ref), or a vector of `DQCType`s for multi-equation
  systems.
- `prob`: SciMLBase-compatible ODE problem.
- `config`: Training configuration, including boundary handling and the scalar
  penalty function.
- `M`: Mesh points used to evaluate the training objective.
- `theta`: Parameters for the variational circuit or circuits. The shape should
  match `DQC`: one parameter vector for a single circuit, or one vector per
  circuit.

# Keyword Arguments

- `optimizer`: Optimisers.jl optimizer used for gradient updates.
- `steps`: Number of training iterations.

# Examples

```julia
using SciMLBase, Yao, QuantumNLDiffEq

prob = ODEProblem((u, p, t) -> -p[1] * u * (p[2] + tan(p[1] * t)),
    [1.0], (0.0, 0.9), [8.0, 0.1])
DQC = [QuantumNLDiffEq.DQCType(
    afm = QuantumNLDiffEq.ChebyshevSparse(2),
    fm = chain(6, [put(i => Ry(0)) for i in 1:6]),
    cost = [Add([put(6, i => Z) for i in 1:6])],
    var = dispatch(EasyBuild.variational_circuit(6, 5), :random),
    N = 6,
)]
config = DQCConfig(abh = QuantumNLDiffEq.Floating(), loss = (a, b) -> abs2(a - b))
M = range(0, stop = 0.9, length = 20)
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
include("precompile.jl")
end
