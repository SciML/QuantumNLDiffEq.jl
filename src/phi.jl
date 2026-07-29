"""
    phi(x, map::AbstractFeatureMap)

Return the scalar feature angle for mesh coordinate `x` and feature map `map`.

# Arguments

- `x`: Scalar mesh coordinate accepted by `map`.
- `map`: Feature-map configuration.

# Returns

Returns the angle encoded by one circuit feature gate.

# Rules

Extensions may define a method only for an [`AbstractFeatureMap`](@ref) subtype
they own. The result must be a scalar accepted by Yao rotation gates for every
coordinate accepted by the corresponding training problem.
"""
function phi(x, ::Product)
    return asin(x)
end

function phi(x, mapping::Union{ChebyshevSparse, ChebyshevTower})
    return mapping.pc * acos(x)
end
