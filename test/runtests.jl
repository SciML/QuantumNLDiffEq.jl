using Test
using SafeTestsets

const GROUP = get(ENV, "GROUP", "All")

if GROUP == "QA"
    import Pkg
    Pkg.activate(joinpath(@__DIR__, "qa"))
    Pkg.instantiate()
    include(joinpath(@__DIR__, "qa", "qa.jl"))
    exit()
end

@safetestset "Tests for damped oscillation equations" begin
    include("damped_oscillation_tests.jl")
end

@safetestset "Encoding multifunction system" begin
    include("encoding_multifunction_tests.jl")
end
