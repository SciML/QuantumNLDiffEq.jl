function calculate_diff_evalue(feature_map_circuit::AbstractBlock, cost::AbstractBlock, var::AbstractBlock, N::Int, theta, x; mapping::FeatureMap)
	diff_evalue = 0.0
	for j in 1:N
		diff_evalue_pos = real.(expect(cost, zero_state(N) => new_circuit(feature_map_circuit, var, x, theta, N, j, π/2; mapping)))

		diff_evalue_neg = real.(expect(cost, zero_state(N) => new_circuit(feature_map_circuit, var, x, theta, N, j, -π/2; mapping)))
				
		diff_evalue += (diff_evalue_pos - diff_evalue_neg) * map_to_circuit(gradient(_x -> phi(_x, mapping), x)[1], j, Product())
	end
	diff_evalue /= 2
	return diff_evalue
end
