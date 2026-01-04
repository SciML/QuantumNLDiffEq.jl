function map_to_circuit(a, ::Real, ::Union{ChebyshevSparse, Product})
    return a
end

function map_to_circuit(a, b::Real, ::ChebyshevTower)
    return a * b
end

function calculate_diff_evalue(DQC::DQCType, cost::Vector{<:AbstractBlock}, theta, x)
    diff_evalue = ComplexF64(0.0)
    for j in 1:DQC.N
        diff_evalue_pos = expect(
            Add(cost), zero_state(DQC.N) => new_circuit(
                DQC, x, theta, j, pi / 2
            )
        )
        diff_evalue_neg = expect(
            Add(cost), zero_state(DQC.N) => new_circuit(
                DQC, x, theta, j, -pi / 2
            )
        )
        diff_evalue += (diff_evalue_pos - diff_evalue_neg) *
            map_to_circuit(gradient(_x -> phi(_x, DQC.afm), x)[1], j, DQC.afm)
    end
    diff_evalue /= 2
    return real(diff_evalue)
end
