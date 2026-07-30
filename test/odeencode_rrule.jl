using ChainRulesCore: rrule
using QuantumNLDiffEq
using SciMLBase: ODEFunction
using Test
using Zygote: gradient

@testset "ODEEncode pullback" begin
    ode = ODEFunction((u, p, t) -> p[1] .* u)
    value, pullback = rrule(QuantumNLDiffEq.ODEEncode, [2.0], [3.0], 0.0, ode)
    tangents = pullback([1.0])

    @test value == [6.0]
    @test tangents[2] == [3.0]
    @test tangents[3] == [2.0]
    @test gradient(u -> sum(QuantumNLDiffEq.ODEEncode(u, [3.0], 0.0, ode)), [2.0])[1] ==
        [3.0]
end
