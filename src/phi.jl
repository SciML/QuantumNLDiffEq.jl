function phi(x, ::Product)
    return asin(x)
end

function phi(x, mapping::Union{ChebyshevSparse, ChebyshevTower})
    return mapping.pc * acos(x)
end
