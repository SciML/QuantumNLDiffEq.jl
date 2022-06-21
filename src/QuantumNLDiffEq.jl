module QuantumNLDiffEq

import Yao: AbstractBlock, zero_state, expect, dispatch!, dispatch, chain
import Flux: update!, ADAM, mse
import Zygote: Buffer, gradient, ignore
import SciMLBase: AbstractODEProblem, AbstractSciMLOperator
	
abstract type AbstractFeatureMap end
abstract type AbstractBoundaryHandling end
abstract type AbstractLoss end
		
struct Product <: AbstractFeatureMap end
struct ChebyshevSparse <: AbstractFeatureMap pc::Int64 end
struct ChebyshevTower <: AbstractFeatureMap pc::Int64 end
struct Pinned <: AbstractBoundaryHandling end
struct Floating <: AbstractBoundaryHandling end

mutable struct DQCType
	afm::AbstractFeatureMap
	abh::AbstractBoundaryHandling
	fm::AbstractBlock
	cost::AbstractBlock
	var::AbstractBlock
	N::Int64
end

include("phi.jl")
include("new_circuit.jl")
include("calculate_diff_evalue.jl")


function loss(DQC::Vector{DQCType}, prob::AbstractODEProblem, M::AbstractVector, theta)
	no_eqns = length(DQC)
	if no_eqns != length(prob.u0)
		throw("Number of encoded equations don't match number of DQCs")
	end
	Loss_diff = 0.0
	for x in 1:length(M)
		evalue = [expect(DQC[i].cost, zero_state(DQC[i].N) => new_circuit(DQC[i], M[x], theta[i])) - expect(DQC[i].cost, zero_state(DQC[i].N) => new_circuit(DQC[i], M[1], theta[i])) + prob.u0[i] for i in 1:no_eqns]

		diff_evalue = [calculate_diff_evalue(DQC[i], theta[i], M[x]) for i in 1:no_eqns]
		
		du = prob.f(zeros(no_eqns), evalue, prob.p, M[x])
		loss_ind = [mse(diff_evalue[i]-du[i], 0) for i in 1:no_eqns]
		
		Loss_diff += sum(loss_ind)
	end
	return real(Loss_diff/length(M))
end

function train!(DQC::Vector{DQCType}, prob::AbstractODEProblem, M::AbstractVector, theta; optimizer=ADAM(0.075), steps=300)
	for _ in 1:steps
		grads = gradient(_theta -> loss(DQC, prob, M, _theta), theta)[1]
		for (p, g) in zip(theta, grads)
  			update!(optimizer, p, g)
		end
		for i in 1:length(DQC)
			dispatch!(DQC[i].var, theta[i])
		end
	end
end

export loss, train!, DQCType
end
