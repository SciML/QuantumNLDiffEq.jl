function new_circuit(feature_map_circuit::AbstractBlock, var::AbstractBlock, x, theta, N::Int, n=1, v=0; mapping::FeatureMap)
	f = repeat([phi(x, mapping)], N)
	ignore() do
		f[n] = f[n] + v
	end

	return chain(N, dispatch(feature_map_circuit, f), dispatch(var, theta))
end
