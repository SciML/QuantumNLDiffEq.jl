# QuantumNLDiffEq.jl

### Installation

```julia
]add https://github.com/VarLad/QuantumNLDiffEq.jl
```

### Usage

```julia

julia> begin
           using Yao, QuantumNLDiffEq
           M = range(start=0; stop=0.9, length=20)
           cost = sum([put(6, i=>Z) for i in 1:6])
           var = dispatch(EasyBuild.variational_circuit(6,5), :random)
           quantum_feature_map_circuit = chain(6, [put(i=>Ry(0)) for i in 1:6])
           params = parameters(var)
           u_0 = 1.0
           N = 6
       end
       
julia> train!(quantum_feature_map_circuit, cost, var, M, N, 1.0, params)

julia> e(M) = [expect(cost, zero_state(N)=>QuantumNLDiffEq.new_circuit(quantum_feature_map_circuit, var, i, params, N, 2; mapping=QuantumNLDiffEq.ChebyshevSparse())) .+ u_0 .- expect(cost, zero_state(N)=>QuantumNLDiffEq.new_circuit(quantum_feature_map_circuit, var, M[1], params, N, 2; mapping=QuantumNLDiffEq.ChebyshevSparse()))  for i in M]

julia> new_M = range(start=0; stop=0.9, length=100)

julia> import Plots

julia> Plots.plot(new_M, real.(e(new_M)), xlabel="x", ylabel="f(x)", legend=false)
```

![example1](https://user-images.githubusercontent.com/51269425/160281466-ced2acc9-fb2e-4ad0-a23a-4bc1dfd934a9.svg)
