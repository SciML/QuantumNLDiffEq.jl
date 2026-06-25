using SciMLTesting, QuantumNLDiffEq, Test
using JET

run_qa(
    QuantumNLDiffEq;
    explicit_imports = true,
    ei_kwargs = (;
        # Optimisers.setup / Optimisers.update are not declared public in Optimisers,
        # but are the documented optimizer-state entry points. Goes away when Optimisers
        # marks them public.
        all_qualified_accesses_are_public = (; ignore = (:setup, :update)),
        # SciMLBase.AbstractODEProblem and ForwardDiff.jacobian are not declared public
        # in their owning packages yet. Drop these once those packages mark them public.
        all_explicit_imports_are_public = (; ignore = (:AbstractODEProblem, :jacobian)),
    ),
)
