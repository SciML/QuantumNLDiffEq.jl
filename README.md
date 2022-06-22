# QuantumNLDiffEq.jl

### Installation

```julia
]add https://github.com/SciML/QuantumNLDiffEq.jl
```

### Usage

```julia
using DifferentialEquations, Yao, QuantumNLDiffEq
# Making the ODEProblem
function f(du, u, p, t)
	λ, κ = p
	du = -1*λ*u*(κ + tan(λ*t))
end
prob = ODEProblem(f, [1.0], (0.0, 0.9), [8.0, 0.1])

#Making the DQC
DQC = [QuantumNLDiffEq.DQCType(QuantumNLDiffEq.ChebyshevSparse(2), QuantumNLDiffEq.Floating(), chain(6, [put(i=>Ry(0)) for i in 1:6]), sum([put(6, i=>Z) for i in 1:6]), dispatch(EasyBuild.variational_circuit(6,5), :random), 6)]
params = [Yao.parameters(DQC[1].var)]

#Training the circuit
QuantumNLDiffEq.train!(DQC, prob, range(start=0; stop=0.9, length=20), params)

#Plotting the solution
using Plots
e(M) = [[expect(DQC[i].cost, zero_state(DQC[i].N) => QuantumNLDiffEq.new_circuit(DQC[i], M[x], params[i])) .- expect(DQC[i].cost, zero_state(DQC[i].N) => QuantumNLDiffEq.new_circuit(DQC[i], M[1], params[i])) .+ prob.u0[i] for i in 1:length(DQC)] for x in 1:length(M)]
new_M = range(start=0; stop=0.9, length=100)
Plots.plot(new_M, reduce(vcat, real.(e(new_M))), xlabel="x", ylabel="f(x)", legend=false)
```

![example1](https://user-images.githubusercontent.com/51269425/168702556-a8e61629-038e-4a9e-be73-7ca8acb4316b.svg)
