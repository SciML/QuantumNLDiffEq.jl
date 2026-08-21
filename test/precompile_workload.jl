using QuantumNLDiffEq
using Test

@testset "Precompile workload" begin
    @test isnothing(QuantumNLDiffEq._precompile_workload())
end
