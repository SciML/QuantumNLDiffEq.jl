function phi(x, ::Product)
	return asin(x)
end

function phi(x, ::Chebyshev)
	return 2*acos(x)
end

