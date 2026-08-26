using PrecompileTools: @compile_workload, @setup_workload

# NOTE: the workload deliberately covers only the forward, non-differentiated
# code path (`phi`, `load`, `new_circuit`, `calculate_evalue`).
#
# `loss` and `train!` must NOT be called here. Both run `Zygote.gradient` on the
# closure `_x -> phi(_x, DQC.afm)` inside `calculate_diff_evalue`. `DQCType.afm`
# is abstractly typed (`::AbstractFeatureMap`), so that closure - and hence the
# `Zygote.Pullback` type whose call operator is `@generated` - is identical for
# every feature map. Running it inside `@compile_workload` bakes the adjoint
# generated for the workload's feature map into the package image, and Julia then
# reuses it for every other feature map, which errors with e.g.
# `BoundsError: attempt to access Tuple{Nothing, Float64} at index [3]` or
# `MethodError: no method matching *(::Float64, ::Nothing)`.
# See test/precompile_workload.jl for the regression test.
function _precompile_workload()
    nqubits = 2
    feature_map = ChebyshevSparse(2)
    feature_circuit = chain(nqubits, [put(i => Ry(0.0)) for i in 1:nqubits])
    observable = Add([put(nqubits, i => Z) for i in 1:nqubits])
    variational_template = EasyBuild.variational_circuit(nqubits, 1)
    variational_circuit = dispatch(variational_template, zeros(nparameters(variational_template)))
    dqc = DQCType(
        afm = feature_map,
        fm = feature_circuit,
        cost = [[observable]],
        var = variational_circuit,
        N = nqubits,
    )
    theta = parameters(dqc.var)
    problem = ODEProblem(
        (u, p, t) -> -p[1] .* u,
        [1.0],
        (0.0, 0.1),
        [0.5],
    )
    config = DQCConfig(abh = Pinned(), loss = (prediction, target) -> abs2(prediction - target))
    mesh = [0.0, 0.1]

    phi(0.25, feature_map)
    load(0.25, nqubits, feature_map)
    new_circuit(dqc, mesh[2], theta)
    calculate_evalue(dqc, dqc.cost[1], problem.u0[1], config.abh, theta, mesh[2], mesh[1])
    calculate_evalue(dqc, dqc.cost[1], problem.u0[1], Floating(), theta, mesh[2], mesh[1])
    ODEEncode([problem.u0[1]], problem.p, mesh[2], problem.f)
    return nothing
end

@setup_workload begin
    @compile_workload begin
        _precompile_workload()
    end
end
