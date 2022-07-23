using QuantumNLDiffEq
using Test
using Yao: dispatch, EasyBuild, put, Z, chain, Ry, parameters, zero_state, expect, parameters, Add
using Zygote: gradient
using DifferentialEquations
using Flux: Adam

#=
@testset "Tests for damped oscillation equations" begin
	function f(u, p, t)
		λ, κ = p
		return -1*λ*u*(κ + tan(λ*t))
	end
	prob = ODEProblem(f, [1.0], (0.0, 0.9), [8.0, 0.1])
	function loss_func(a, b)
		return (a - b)^2
	end
	
	@testset "Boundary Handling Tests" begin
		DQC = [QuantumNLDiffEq.DQCType(afm = QuantumNLDiffEq.ChebyshevSparse(2), fm = chain(6, [put(i=>Ry(0)) for i in 1:6]), cost = [Add([put(6, i=>Z) for i in 1:6])], var = dispatch(EasyBuild.variational_circuit(6,5), :random), N = 6)]
		config(boundary) = DQCConfig(abh = boundary, loss = loss_func)
		M = range(start=0; stop=0.9, length=20)
		evalue(M, conf) = [QuantumNLDiffEq.calculate_evalue(DQC[1], DQC[1].cost, prob.u0[1], conf.abh, params[1], M[x], M[1]) for x in 1:length(M)]
		
		@testset "Test for Pinned Boundary Handling" begin
			conf = config(QuantumNLDiffEq.Pinned(2.5))
			params = [parameters(DQC[1].var)]
			QuantumNLDiffEq.train!(DQC, prob, conf, M, params)
			@test QuantumNLDiffEq.loss(DQC, prob, conf, M, params) < 0.5
		end
		
		@testset "Test for Floating Boundary Handling" begin
			conf = config(QuantumNLDiffEq.Floating())
			params = [parameters(DQC[1].var)]
			QuantumNLDiffEq.train!(DQC, prob, conf, M, params)
			@test QuantumNLDiffEq.loss(DQC, prob, conf, M, params) < 0.5
		end
		
		@testset "Test for Optimised Boundary Handling" begin
			conf = config(QuantumNLDiffEq.Optimized(rand()))
			params = [parameters(DQC[1].var)]
			QuantumNLDiffEq.train!(DQC, prob, conf, M, params)
				@test QuantumNLDiffEq.loss(DQC, prob, conf, M, params) < 0.5
		end
	end
	
	@testset "Mapping Tests" begin
		DQ(mapping) = [QuantumNLDiffEq.DQCType(afm = mapping, fm = chain(6, [put(i=>Ry(0)) for i in 1:6]), cost = [Add([put(6, i=>Z) for i in 1:6])], var = dispatch(EasyBuild.variational_circuit(6,5), :random), N = 6)]
		config = DQCConfig(abh = QuantumNLDiffEq.Floating(), loss = loss_func)
		M = range(start=0; stop=0.9, length=20)
		evalue(M, mapping) = [QuantumNLDiffEq.calculate_evalue(DQC(mapping), DQC(mapping)[1].cost, prob.u0[1], conf.abh, params[1], M[x], M[1]) for x in 1:length(M)]
		
		@testset "Test for Product Feature Mapping" begin
			input = DQ(QuantumNLDiffEq.Product())
			params = [parameters(input[1].var)]
			QuantumNLDiffEq.train!(input, prob, config, M, params)
			@test QuantumNLDiffEq.loss(input, prob, config, M, params) < 10
		end
		
		@testset "Test for Chebyshev Sparse Mapping" begin
			input = QuantumNLDiffEq.DQCType(afm = QuantumNLDiffEq.ChebyshevSparse(2), fm = chain(6, [put(i=>Ry(0)) for i in 1:6]), cost = [[Add([put(6, i=>Z) for i in 1:6])]], var = dispatch(EasyBuild.variational_circuit(6,5), :random), N = 6)
			params = parameters(input.var)
			QuantumNLDiffEq.train!(input, prob, config, M, params)
			@test QuantumNLDiffEq.loss(input, prob, config, M, params) < 0.5
		end
		
		@testset "Test for Chebyshev Tower Mapping" begin
			input = DQ(QuantumNLDiffEq.ChebyshevTower(2))
			params = [parameters(input[1].var)]
			QuantumNLDiffEq.train!(input, prob, config, M, params)
			@test QuantumNLDiffEq.loss(input, prob, config, M, params) < 0.5
		end
	end
	#
	@testset "Regularization Tests" begin
		@testset "Singlular Encoding" begin
			DQC = QuantumNLDiffEq.DQCType(afm = QuantumNLDiffEq.ChebyshevSparse(2), fm = chain(6, [put(i=>Ry(0)) for i in 1:6]), cost = [[Add([put(6, i=>Z) for i in 1:6])]], var = dispatch(EasyBuild.variational_circuit(6,5), :random), N = 6)
			config = DQCConfig(reg = QuantumNLDiffEq.RegularisationParams([[1.0, 0.04724630684751344, -0.7340180576454999, -0.10424985483835963, 0.5322086377394599]], [0.0, 0.18947368421052632, 0.37894736842105264, 0.5684210526315789, 0.7578947368421053], 0.0),  abh = QuantumNLDiffEq.Floating(), loss = loss_func)
			M = range(start=0; stop=0.9, length=20)
			evalue(M) = [QuantumNLDiffEq.calculate_evalue(QuantumNLDiffEqChebyshevSparse(2), DQC(mapping)[1].cost, prob.u0[1], config.abh, params[1], M[x], M[1]) for x in 1:length(M)]
			params = parameters(DQC.var)
			QuantumNLDiffEq.tr_custom!(DQC, prob, config, M, params)
			@show loss = QuantumNLDiffEq.loss(DQC, prob, config, M, params)
			@test loss < 0.5
		end
		
		@testset "Multiple Encoding" begin
			DQC = [QuantumNLDiffEq.DQCType(afm = QuantumNLDiffEq.ChebyshevSparse(2), fm = chain(6, [put(i=>Ry(0)) for i in 1:6]), cost = [Add([put(6, i=>Z) for i in 1:6])], var = dispatch(EasyBuild.variational_circuit(6,5), :random), N = 6)]
			config = DQCConfig(reg = QuantumNLDiffEq.RegularisationParams([[1.0, 0.04724630684751344, -0.7340180576454999, -0.10424985483835963, 0.5322086377394599]], [0.0, 0.18947368421052632, 0.37894736842105264, 0.5684210526315789, 0.7578947368421053], 0.0), abh = QuantumNLDiffEq.Floating(), loss = loss_func)
			M = range(start=0; stop=0.9, length=20)
			evalue(M) = [QuantumNLDiffEq.calculate_evalue(QuantumNLDiffEqChebyshevSparse(2), DQC(mapping)[1].cost, prob.u0[1], config.abh, params[1], M[x], M[1]) for x in 1:length(M)]
			params = [parameters(DQC[1].var)]
			QuantumNLDiffEq.tr_custom!(DQC, prob, config, M, params)
			loss = QuantumNLDiffEq.loss(DQC, prob, config, M, params)
			@test loss < 0.5
		end
	end
end
=#

@testset "Encoding multifunction system" begin
	function f(u, p, t)
		λ1, λ2 = p
		return [λ1*u[2] + λ2*u[1], -λ1*u[1] - λ2*u[2]]
	end
	prob = ODEProblem(f, [0.5, 0.0], (0.0, 0.9), [5.0, 3.0])
	function loss_func(a, b)
		return (a - b)^2
	end
	config = DQCConfig(abh = QuantumNLDiffEq.Floating(), loss = loss_func)
	DQC = repeat(QuantumNLDiffEq.DQCType(afm = QuantumNLDiffEq.ChebyshevTower(2), fm = chain(6, [put(i=>Ry(0)) for i in 1:6]), cost = [sum([put(6, i=>Z) for i in 1:6])], var = dispatch(EasyBuild.variational_circuit(6,5), :random), N = 6), 2)
	params = [Yao.parameters(DQC[1].var), Yao.parameters(DQC[2].var)]
	M = range(start=0; stop=0.9, length=20)
	QuantumNLDiffEq.train!(DQC, prob, config, M, params; optimizer=Adam(0.02) steps = 400)
	@test QuantumNLDiffEq.loss(DQC, prob, config, M, params) < 0.05
end