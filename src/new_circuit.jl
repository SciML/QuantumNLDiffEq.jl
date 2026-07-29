"""
    load(x, N, map::AbstractFeatureMap)

Return the feature angles encoded on the `N` qubits of a Differential Quantum
Circuit at mesh coordinate `x`.

# Arguments

- `x`: Scalar mesh coordinate.
- `N`: Positive number of circuit qubits.
- `map`: Feature-map configuration.

# Returns

Returns an indexable collection containing exactly `N` angles.

# Rules

Extensions may define a method only for an [`AbstractFeatureMap`](@ref) subtype
they own. The collection length must equal `N`, and every element must be a
valid Yao rotation angle.
"""
function load(x, N, mapping::Product)
    return repeat([phi(x, mapping)], N)
end

function load(x, N, mapping::ChebyshevSparse)
    return repeat([phi(x, mapping)], N)
end

function load(x, N, mapping::ChebyshevTower)
    return [i * phi(x, mapping) for i in 1:N]
end

function new_circuit(DQC::DQCType, x, theta, n = 1, v = 0)
    tmp = load(x, DQC.N, DQC.afm)
    f = [i == n ? tmp[i] .+ v : tmp[i] for i in 1:DQC.N]

    return chain(DQC.N, dispatch(DQC.fm, f), DQC.evol, dispatch(DQC.var, theta))
end
