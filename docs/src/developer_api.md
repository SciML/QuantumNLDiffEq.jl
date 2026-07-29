# Developer API

The names on this page are versioned extension points for packages that own a
feature-map or boundary-handling implementation. They are not intended for
ordinary application code. Applications should use the concrete configurations
on the public API page and call [`loss`](@ref) or [`train!`](@ref).

## Feature Maps

```@docs
QuantumNLDiffEq.AbstractFeatureMap
QuantumNLDiffEq.phi
QuantumNLDiffEq.load
QuantumNLDiffEq.map_to_circuit
```

## Boundary Handling

```@docs
QuantumNLDiffEq.AbstractBoundaryHandling
QuantumNLDiffEq.calculate_evalue
QuantumNLDiffEq.loss_bound
```
