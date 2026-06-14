using QuantumNLDiffEq
using Yao: dispatch, EasyBuild, put, Z, chain, Ry, parameters, zero_state, expect,
    parameters, nparameters, Add
using Zygote: gradient
using DifferentialEquations
using Optimisers: Adam
using Random
using Test

# Generate reproducible initial parameters using a local Xoshiro RNG so the
# variational-circuit initialization is identical across Julia versions
# (Julia's default global RNG stream changes between releases, which made
# the loss-threshold tests below flaky when using `dispatch(..., :random)`).
function init_var_circuit(seed::Integer)
    circ = EasyBuild.variational_circuit(6, 5)
    rng = Xoshiro(seed)
    return dispatch(circ, rand(rng, nparameters(circ)))
end

M = range(0; stop = 0.9, length = 20)

function f(u, p, t)
    λ1, λ2 = p
    return [λ1 * u[2] + λ2 * u[1], -λ1 * u[1] - λ2 * u[2]]
end
prob = ODEProblem(f, [0.5, 0.0], (0.0, 0.9), [5.0, 3.0])
function loss_func(a, b)
    return (a - b)^2
end
config = DQCConfig(abh = QuantumNLDiffEq.Floating(), loss = loss_func)
DQC = repeat(
    [
        QuantumNLDiffEq.DQCType(
            afm = QuantumNLDiffEq.ChebyshevTower(2),
            fm = chain(6, [put(i => Ry(0)) for i in 1:6]),
            cost = [sum([put(6, i => Z) for i in 1:6])],
            var = init_var_circuit(42), N = 6
        ),
    ],
    2
)
params = [parameters(DQC[1].var), parameters(DQC[2].var)]
QuantumNLDiffEq.train!(
    DQC, prob, config, M, params; optimizer = Adam(0.02), steps = 400
)
@test QuantumNLDiffEq.loss(DQC, prob, config, M, params) < 0.05
