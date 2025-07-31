function calculate_evalue(DQC::DQCType, cost::Vector{<:AbstractBlock}, u0::Float64,
        ::Floating, theta, M_value::Real, M_initial::Real)
    return expect(Add(cost), zero_state(DQC.N) => new_circuit(DQC, M_value, theta)) -
           expect(Add(cost), zero_state(DQC.N) => new_circuit(DQC, M_initial, theta)) + u0
end

function calculate_evalue(DQC::DQCType, cost::Vector{<:AbstractBlock},
        ::Float64, ::Pinned, theta, M_value::Real, ::Real)
    return expect(Add(cost), zero_state(DQC.N) => new_circuit(DQC, M_value, theta))
end

function calculate_evalue(DQC::DQCType, cost::Vector{<:AbstractBlock}, ::Float64,
        abh::Optimized, theta, M_value::Real, ::Real)
    return abh.fc + expect(Add(cost), zero_state(DQC.N) => new_circuit(DQC, M_value, theta))
end
