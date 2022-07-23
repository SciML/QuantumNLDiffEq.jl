# QuantumNLDiffEq.jl

### Installation

```julia
]add https://github.com/SciML/QuantumNLDiffEq.jl
```

### Usage

```julia
using DifferentialEquations, Yao, QuantumNLDiffEq
# Making the ODEProblem
function f(u, p, t)
	λ, κ = p
	return -1*λ*u*(κ + tan(λ*t))
end
prob = ODEProblem(f, [1.0], (0.0, 0.9), [8.0, 0.1])
function loss_func(a, b)
	return (a - b)^2
end

#Making the DQC
DQC = [QuantumNLDiffEq.DQCType(afm = QuantumNLDiffEq.ChebyshevSparse(2), fm = chain(6, [put(i=>Ry(0)) for i in 1:6]), cost = [Add([put(6, i=>Z) for i in 1:6])], var = dispatch(EasyBuild.variational_circuit(6,5), :random), N = 6)]
config = DQCConfig(abh = QuantumNLDiffEq.Floating(), loss = loss_func)
M = range(start=0; stop=0.9, length=20)
evalue(M) = [QuantumNLDiffEq.calculate_evalue(DQC[1], DQC[1].cost, prob.u0[1], conf.abh, params[1], M[x], M[1]) for x in 1:length(M)]
params = [Yao.parameters(DQC[1].var)]

#Training the circuit
QuantumNLDiffEq.train!(DQC, prob, conf, M, params)

#Plotting the solution
using Plots
new_M = range(start=0; stop=0.9, length=100)
Plots.plot(new_M, reduce(vcat, real.(evalue(new_M))), xlabel="x", ylabel="f(x)", legend=false)
```

![example1](https://user-images.githubusercontent.com/51269425/168702556-a8e61629-038e-4a9e-be73-7ca8acb4316b.svg)
