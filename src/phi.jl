function phi(x, ::Real, ::Product)
	return asin(x)
end

function phi(x, b::Real, ::Union{ChebyshevSparse, ChebyshevTower})
	return b*acos(x)
end

