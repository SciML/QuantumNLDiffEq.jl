using QuantumNLDiffEq
using Aqua
using JET
using Test

@testset "Aqua" begin
    Aqua.test_all(QuantumNLDiffEq)
end

@testset "JET" begin
    JET.test_package(QuantumNLDiffEq; target_defined_modules = true)
end
