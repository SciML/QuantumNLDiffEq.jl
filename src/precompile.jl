using PrecompileTools: @compile_workload, @setup_workload

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
    calculate_evalue(dqc, dqc.cost[1], problem.u0[1], config.abh, theta, mesh[2], mesh[1])
    loss(dqc, problem, config, mesh, theta)
    train!(dqc, problem, config, mesh, theta; steps = 1)
    return nothing
end

@setup_workload begin
    @compile_workload begin
        _precompile_workload()
    end
end
