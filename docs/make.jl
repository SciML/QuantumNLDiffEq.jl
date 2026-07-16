using Pkg
Pkg.develop(PackageSpec(path = dirname(@__DIR__)))

using Documenter
using QuantumNLDiffEq

makedocs(;
    modules = [QuantumNLDiffEq],
    sitename = "QuantumNLDiffEq.jl",
    pages = [
        "Home" => "index.md",
    ],
    clean = true,
    doctest = false,
    checkdocs = :exports,
)

deploydocs(; repo = "github.com/SciML/QuantumNLDiffEq.jl.git", push_preview = true)
