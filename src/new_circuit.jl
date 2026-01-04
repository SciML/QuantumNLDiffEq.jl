function load(x, N, mapping::Product)
    return repeat([phi(x, mapping)], N)
end

function load(x, N, mapping::ChebyshevSparse)
    return repeat([phi(x, mapping)], N)
end

function load(x, N, mapping::ChebyshevTower)
    return [i * phi(x, mapping) for i in 1:N]
end

function new_circuit(DQC::DQCType, x, theta, n = 1, v = 0)
    tmp = load(x, DQC.N, DQC.afm)
    f = [i == n ? tmp[i] .+ v : tmp[i] for i in 1:DQC.N]

    return chain(DQC.N, dispatch(DQC.fm, f), DQC.evol, dispatch(DQC.var, theta))
end
