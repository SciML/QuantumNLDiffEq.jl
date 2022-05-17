using QuantumNLDiffEq
using Test
using Yao: dispatch, EasyBuild, put, Z, chain, Ry, parameters, zero_state, expect
using Zygote: gradient

≃(a, b) = round(a, digits = 2) ==  round(b, digits = 2)

@testset "Check if chainrules_patch works" begin
	N = 6	
	cost = sum([put(N, i=>Z) for i in 1:N])
	theta = rand(96)
	new_var = dispatch(EasyBuild.variational_circuit(N,5), :random);
	function new_loss(theta, j)
		new_circ = dispatch(new_var, theta)
		diff_evalue = expect(cost, zero_state(N) => new_circ)
		return real.(diff_evalue)*j
	end
	
	p = gradient(_theta->new_loss(_theta, 1), theta)[1][3]
	d = gradient(_theta->new_loss(_theta, 3), theta)[1][3]
	@test d ≃ p*3
end

@testset "Tests for Chebyshev sparse mapping" begin
	M = range(0, stop=0.9, length=20)
        cost = sum([put(6, i=>Z) for i in 1:6])
        var = dispatch(EasyBuild.variational_circuit(6,5), :random)
        quantum_feature_map_circuit = chain(6, [put(i=>Ry(0)) for i in 1:6])
        params = parameters(var)
        u_0 = 1.0
        N = 6
        train!(quantum_feature_map_circuit, cost, var, M, N, 1.0, params)
         
	@testset "Check if FD works" begin
		t = 6
		evalue(x) = expect(cost, zero_state(N)=>QuantumNLDiffEq.new_circuit(quantum_feature_map_circuit, var, x, params, N, 2; mapping=QuantumNLDiffEq.ChebyshevSparse()))

		x_fd = M[t]
		dx = 0.0001
		x_fd += dx
		loss_grad_fd = (evalue(x_fd) - evalue(M[t]))/dx

		@test QuantumNLDiffEq.calculate_diff_evalue(quantum_feature_map_circuit, cost, var, N, params, M[t], 2; mapping=QuantumNLDiffEq.ChebyshevSparse()) ≃ loss_grad_fd
	end
	
	@test loss(quantum_feature_map_circuit, cost, var, M, N, u_0, params, 2; mapping=QuantumNLDiffEq.ChebyshevSparse()) < 1.0
end
