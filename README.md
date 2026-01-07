# QuantumNLDiffEq.jl

QuantumNLDiffEq.jl is a package for solving nonlinear differential equations using Differential Quantum Circuits (DQCs). It integrates with the SciML ecosystem to leverage quantum computing approaches for differential equation solving.

## Installation

```julia
]add https://github.com/SciML/QuantumNLDiffEq.jl
```

## Quick Start

```julia
using DifferentialEquations, Yao, QuantumNLDiffEq

# Define the ODE problem
function f(u, p, t)
    λ, κ = p
    return -1*λ*u*(κ + tan(λ*t))
end
prob = ODEProblem(f, [1.0], (0.0, 0.9), [8.0, 0.1])

# Define the loss function for training
function loss_func(a, b)
    return (a - b)^2
end

# Create the Differential Quantum Circuit
DQC = [QuantumNLDiffEq.DQCType(
    afm = QuantumNLDiffEq.ChebyshevSparse(2),  # Chebyshev polynomial feature mapping
    fm = chain(6, [put(i=>Ry(0)) for i in 1:6]),  # Feature map circuit
    cost = [Add([put(6, i=>Z) for i in 1:6])],  # Cost function (observable)
    var = dispatch(EasyBuild.variational_circuit(6, 5), :random),  # Variational circuit
    N = 6  # Number of qubits
)]

# Configure the training
config = DQCConfig(abh = QuantumNLDiffEq.Floating(), loss = loss_func)
M = range(start=0, stop=0.9, length=20)  # Mesh points for training
params = [Yao.parameters(DQC[1].var)]

# Train the quantum circuit to solve the ODE
QuantumNLDiffEq.train!(DQC, prob, config, M, params)

# Evaluate and plot the solution
evalue(M) = [QuantumNLDiffEq.calculate_evalue(DQC[1], DQC[1].cost, prob.u0[1],
                                               config.abh, params[1], M[x], M[1])
             for x in 1:length(M)]

using Plots
new_M = range(start=0, stop=0.9, length=100)
plot(new_M, reduce(vcat, real.(evalue(new_M))), xlabel="x", ylabel="f(x)", legend=false)
```

![example1](https://user-images.githubusercontent.com/51269425/180599519-4e29b5c0-36e9-497b-b63c-db97d14a1050.png)
