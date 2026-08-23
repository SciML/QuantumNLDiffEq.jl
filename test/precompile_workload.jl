using QuantumNLDiffEq
using Yao: Add, EasyBuild, Ry, Z, chain, dispatch, nparameters, parameters, put
using SciMLBase: ODEProblem
using Test

@testset "Precompile workload" begin
    @test isnothing(QuantumNLDiffEq._precompile_workload())
end

# Regression test for the precompile workload corrupting Zygote's cached
# adjoints. `DQCType.afm` is abstractly typed, so the closure differentiated
# inside `calculate_diff_evalue` has the same type for every feature map. When
# the precompile workload ran `loss`/`train!`, the `@generated` Zygote pullback
# compiled for the workload's feature map was baked into the package image and
# reused for all other feature maps, so `loss` and `train!` errored for every
# feature map except the one exercised by the workload.
@testset "Differentiation works for every feature map" begin
    nqubits = 2
    mesh = range(0; stop = 0.3, length = 3)
    problem = ODEProblem((u, p, t) -> -p[1] .* u, [1.0], (0.0, 0.3), [0.5])
    config = DQCConfig(
        abh = QuantumNLDiffEq.Floating(), loss = (a, b) -> (a - b)^2
    )

    function build(mapping)
        template = EasyBuild.variational_circuit(nqubits, 1)
        return [
            QuantumNLDiffEq.DQCType(
                afm = mapping,
                fm = chain(nqubits, [put(i => Ry(0.0)) for i in 1:nqubits]),
                cost = [Add([put(nqubits, i => Z) for i in 1:nqubits])],
                var = dispatch(template, zeros(nparameters(template))),
                N = nqubits
            ),
        ]
    end

    mappings = (
        "Product" => QuantumNLDiffEq.Product(),
        "ChebyshevSparse" => QuantumNLDiffEq.ChebyshevSparse(2),
        "ChebyshevTower" => QuantumNLDiffEq.ChebyshevTower(2),
    )

    for (name, mapping) in mappings
        @testset "$name" begin
            dqc = build(mapping)
            params = [parameters(dqc[1].var)]
            @test isfinite(QuantumNLDiffEq.loss(dqc, problem, config, mesh, params))
            QuantumNLDiffEq.train!(dqc, problem, config, mesh, params; steps = 1)
            @test isfinite(QuantumNLDiffEq.loss(dqc, problem, config, mesh, params))
        end
    end
end
