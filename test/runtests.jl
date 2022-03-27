using QuantumNLDiffEq
using Test
using Yao: dispatch, EasyBuild, put, Z, chain, Ry, parameters, zero_state, expect
using Zygote: gradient

@testset "Match trained equation" begin
	M = range(0, stop=0.9, length=20)
        cost = sum([put(6, i=>Z) for i in 1:6])
        var = dispatch(EasyBuild.variational_circuit(6,5), :random)
        quantum_feature_map_circuit = chain(6, [put(i=>Ry(0)) for i in 1:6])
        params = parameters(var)
        u_0 = 1.0
        N = 6
        train!(quantum_feature_map_circuit, cost, var, M, N, 1.0, params)
        
         e = real.([expect(cost, zero_state(N)=>QuantumNLDiffEq.new_circuit(quantum_feature_map_circuit, var, i, params, N; mapping=QuantumNLDiffEq.Chebyshev())) .+ u_0 .- expect(cost, zero_state(N)=>QuantumNLDiffEq.new_circuit(quantum_feature_map_circuit, var, M[1], params, N; mapping=QuantumNLDiffEq.Chebyshev()))  for i in M])
         
	≃(a, b) = round(a, digits = 2) ==  round(b, digits = 2)
	@test e[1] ≃ 1.0
	@test e[2] ≃ 0.83986
	@test e[6] ≃ -0.236077
	@test e[10] ≃ -0.657542
	@test e[14] ≃ 0.134493
	@test e[20] ≃ 0.305012
end

