using QuantumNLDiffEq
using Test
using Yao: Add, EasyBuild, Ry, Z, chain, dispatch, put

struct QuadraticFeatureMap <: QuantumNLDiffEq.AbstractFeatureMap
    scale::Float64
end

QuantumNLDiffEq.phi(x, map::QuadraticFeatureMap) = map.scale * x^2
QuantumNLDiffEq.load(x, N, map::QuadraticFeatureMap) = fill(QuantumNLDiffEq.phi(x, map), N)
QuantumNLDiffEq.map_to_circuit(a, j::Real, ::QuadraticFeatureMap) = a * j

struct OffsetBoundary <: QuantumNLDiffEq.AbstractBoundaryHandling
    offset::Float64
end

function QuantumNLDiffEq.calculate_evalue(
        ::QuantumNLDiffEq.DQCType, cost, u0, boundary::OffsetBoundary, theta,
        M_value, M_initial
    )
    return u0 + boundary.offset
end

function QuantumNLDiffEq.loss_bound(
        DQC, M, u0, cost_params, ::OffsetBoundary, loss, theta
    )
    return 0.0
end

@testset "Public extension interfaces" begin
    map = QuadraticFeatureMap(2.0)
    @test QuantumNLDiffEq.phi(0.5, map) == 0.5
    @test QuantumNLDiffEq.load(0.5, 3, map) == fill(0.5, 3)
    @test QuantumNLDiffEq.map_to_circuit(0.5, 2, map) == 1.0

    dqc = QuantumNLDiffEq.DQCType(
        afm = map,
        fm = chain(2, [put(i => Ry(0.0)) for i in 1:2]),
        cost = [Add([put(2, i => Z) for i in 1:2])],
        var = dispatch(EasyBuild.variational_circuit(2, 1), :zero),
        N = 2,
    )
    boundary = OffsetBoundary(0.25)
    config = QuantumNLDiffEq.DQCConfig(
        abh = boundary,
        loss = (prediction, target) -> abs2(prediction - target),
    )

    @test config.abh === boundary
    @test QuantumNLDiffEq.calculate_evalue(
        dqc, dqc.cost, 1.0, boundary, [0.0], 0.5, 0.0
    ) == 1.25
    @test QuantumNLDiffEq.loss_bound(
        dqc, 0.0, [1.0], QuantumNLDiffEq.NoCostParams(), boundary,
        config.loss, [0.0]
    ) == 0.0
end
