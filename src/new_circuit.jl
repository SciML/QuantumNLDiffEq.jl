function load(x, pc, N, mapping::Union{Product,ChebyshevSparse})
	return repeat([phi(x, pc, mapping)], N)
end

function load(x, pc, N, mapping::ChebyshevTower)
	return [phi(x, pc*i, mapping) for i in 1:N]
end

function new_circuit(feature_map_circuit::AbstractBlock, var::AbstractBlock, x, theta, N::Int, pc, n=1, v=0; mapping::FeatureMap)
	tmp = load(x, pc, N, mapping)
    f = [i == n ? tmp[i] .+ v : tmp[i] for i in 1:N]

	return chain(N, dispatch(feature_map_circuit, f), dispatch(var, theta))
end
