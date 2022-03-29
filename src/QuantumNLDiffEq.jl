module QuantumNLDiffEq

import Yao: AbstractBlock, zero_state, expect, dispatch!, dispatch, chain
import Flux: update!, ADAM, mse
import Zygote: Buffer, gradient, ignore
	
abstract type FeatureMap end
abstract type BoundaryHandling end
	
struct Product <: FeatureMap end
struct Chebyshev <: FeatureMap end
struct Pinned <: BoundaryHandling end
struct Floating <: BoundaryHandling end
		
function map_to_circuit(a, b, ::Chebyshev)
	return a * b
end

function map_to_circuit(a, b, ::Product)
	return a
end
	
include("phi.jl")
include("new_circuit.jl")
include("calculate_diff_evalue.jl")
	
loss_calc(diff_evalue, evalue, x) = mse(diff_evalue + 8*evalue*(0.1 + tan(8*x)), 0)
	
function loss(feature_map_circuit::AbstractBlock, cost::AbstractBlock, var::AbstractBlock, M::AbstractVector, N::Int, u_0::Real, theta; mapping::FeatureMap)
	Loss_diff = 0.0
	for x in 1:length(M)
		evalue = expect(cost, zero_state(N)=>new_circuit(feature_map_circuit, var, M[x], theta, N; mapping)) - expect(cost, zero_state(N)=>new_circuit(feature_map_circuit, var, M[1], theta, N; mapping)) + u_0
			
		diff_evalue = calculate_diff_evalue(feature_map_circuit, cost, var, N, theta, M[x]; mapping)

		Loss_diff += loss_calc(diff_evalue, evalue, M[x])
	end	
	return real.(Loss_diff/length(M))
end
	
function train!(feature_map_circuit::AbstractBlock, cost::AbstractBlock, var::AbstractBlock, M::AbstractVector, N::Int, u_0::Real, theta; mapping::FeatureMap=Chebyshev(), boundary::BoundaryHandling=Floating(), optimizer=ADAM(0.05), steps=300)
	for i in 1:steps
		grads = gradient(_theta->loss(feature_map_circuit, cost, var, M, N, u_0, _theta; mapping), theta)[1]
		update!(optimizer, theta, grads)
		dispatch!(var, theta)
	end
end
	
export loss, train!

end
