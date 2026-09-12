using QuantumNLDiffEq, BenchmarkTools
using Yao: dispatch, EasyBuild, put, Z, chain, Ry, parameters, nparameters, Add
using SciMLBase
using Random

const SUITE = BenchmarkGroup()

# Reproducible variational circuit (same pattern as the package tests)
function init_var_circuit(seed::Integer)
    circ = EasyBuild.variational_circuit(6, 5)
    rng = Xoshiro(seed)
    return dispatch(circ, rand(rng, nparameters(circ)))
end

M = range(0; stop = 0.9, length = 20)

f(u, p, t) = -1 * p[1] * u * (p[2] + tan(p[1] * t))
prob = ODEProblem(f, [1.0], (0.0, 0.9), [8.0, 0.1])
loss_func(a, b) = (a - b)^2

dqc = QuantumNLDiffEq.DQCType(
    afm = QuantumNLDiffEq.ChebyshevSparse(2),
    fm = chain(6, [put(i => Ry(0)) for i in 1:6]),
    cost = [Add([put(6, i => Z) for i in 1:6])],
    var = init_var_circuit(42), N = 6
)
conf = DQCConfig(abh = QuantumNLDiffEq.Pinned(2.5), loss = loss_func)
params = [parameters(dqc.var)]

# =============================================================================
# DQC construction
# =============================================================================

SUITE["construct"] = BenchmarkGroup()

SUITE["construct"]["dqc_type"] = @benchmarkable QuantumNLDiffEq.DQCType(
    afm = QuantumNLDiffEq.ChebyshevSparse(2),
    fm = chain(6, [put(i => Ry(0)) for i in 1:6]),
    cost = [Add([put(6, i => Z) for i in 1:6])],
    var = init_var_circuit(42), N = 6
)
SUITE["construct"]["dqc_config"] = @benchmarkable DQCConfig(
    abh = QuantumNLDiffEq.Pinned(2.5), loss = $loss_func
)
SUITE["construct"]["var_circuit"] = @benchmarkable init_var_circuit(42)

# =============================================================================
# Loss evaluation + short training run
# =============================================================================

SUITE["train"] = BenchmarkGroup()

SUITE["train"]["loss"] = @benchmarkable QuantumNLDiffEq.loss(
    $([dqc]), $prob, $conf, $M, $params
)
SUITE["train"]["train_20steps"] = @benchmarkable QuantumNLDiffEq.train!(
    $([dqc]), $prob, $conf, $M, $params; steps = 20
)
