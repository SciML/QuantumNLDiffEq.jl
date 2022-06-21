using QuantumNLDiffEq
using Test
using Yao: dispatch, EasyBuild, put, Z, chain, Ry, parameters, zero_state, expect
using Zygote: gradient
using DifferentialEquations

@testset "Tests for Chebyshev sparse mapping" begin
	function f(du, u, p, t)
		λ, κ = p
		du = -1*λ*u*(κ + tan(λ*t))
	end
	prob = ODEProblem(f, [1.0], (0.0, 0.9), [8.0, 0.1])
	DQC = [QuantumNLDiffEq.DQCType(QuantumNLDiffEq.ChebyshevSparse(2), QuantumNLDiffEq.Floating(), chain(6, [put(i=>Ry(0)) for i in 1:6]), sum([put(6, i=>Z) for i in 1:6]), dispatch(EasyBuild.variational_circuit(6,5), :random), 6)]
	params = [parameters(DQC[1].var)]
	QuantumNLDiffEq.train!(DQC, prob, range(0, 0.9, length=20), params)
	@test QuantumNLDiffEq.loss(DQC, prob, range(0, 0.9, length=20), params) < 1.0
end
