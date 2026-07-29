"""
    calculate_evalue(DQC, cost, u0, boundary, theta, M_value, M_initial)

Return the state prediction represented by `DQC` at mesh point `M_value`.

# Arguments

- `DQC`: Differential Quantum Circuit whose observable is evaluated.
- `cost`: Cost-observable blocks for the encoded state component.
- `u0`: Initial state value for that component.
- `boundary`: Boundary-handling strategy.
- `theta`: Current variational-circuit parameters.
- `M_value`: Mesh point where the state is evaluated.
- `M_initial`: First mesh point of the training interval.

# Returns

Returns the scalar circuit prediction adjusted according to `boundary`.

# Rules

Extensions may define methods only for an [`AbstractBoundaryHandling`](@ref)
subtype they own. A method must return a real-valued state prediction and must
not mutate `DQC`, `cost`, `theta`, or either mesh coordinate.
"""
function calculate_evalue(
        DQC::DQCType, cost::Vector{<:AbstractBlock}, u0::Float64,
        ::Floating, theta, M_value::Real, M_initial::Real
    )
    return expect(Add(cost), zero_state(DQC.N) => new_circuit(DQC, M_value, theta)) -
        expect(Add(cost), zero_state(DQC.N) => new_circuit(DQC, M_initial, theta)) + u0
end

function calculate_evalue(
        DQC::DQCType, cost::Vector{<:AbstractBlock},
        ::Float64, ::Pinned, theta, M_value::Real, ::Real
    )
    return expect(Add(cost), zero_state(DQC.N) => new_circuit(DQC, M_value, theta))
end

function calculate_evalue(
        DQC::DQCType, cost::Vector{<:AbstractBlock}, ::Float64,
        abh::Optimized, theta, M_value::Real, ::Real
    )
    return abh.fc + expect(Add(cost), zero_state(DQC.N) => new_circuit(DQC, M_value, theta))
end
