using SciMLTesting, QuantumNLDiffEq, Test
using JET

run_qa(
    QuantumNLDiffEq;
    explicit_imports = true,
    ei_kwargs = (;
        # ForwardDiff.jacobian is not declared public in ForwardDiff yet. Drop this once
        # ForwardDiff marks it public.
        all_explicit_imports_are_public = (; ignore = (:jacobian,)),
    ),
)
