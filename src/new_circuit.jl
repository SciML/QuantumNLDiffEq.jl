function new_circuit(feature_map_circuit::AbstractBlock, var::AbstractBlock, x, theta, N::Int, n=1, v=0; mapping::FeatureMap)
	tmp = repeat([phi(x, mapping)], N)
    	f = [i == n ? tmp[i] .+ v : tmp[i] for i in 1:N]

	return chain(N, dispatch(feature_map_circuit, f), dispatch(var, theta))
end
