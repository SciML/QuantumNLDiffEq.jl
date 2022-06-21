function phi(x, ::Product)
	return asin(x)
end

function phi(x, map::Union{ChebyshevSparse, ChebyshevTower})
	return map.pc*acos(x)
end

