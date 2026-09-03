# Contribution workflow

Scientific changes are easiest to review when each concern is changed in its natural owner and the evidence is specified before implementation grows around it.

## Start with the contract

For a behavioral change, write down:

1. the definition being changed or added;
2. the representation that carries it;
3. the invariant that distinguishes correct from incorrect behavior;
4. the oracle that will evaluate the invariant;
5. the environment required to obtain evidence.

Then implement the smallest change that satisfies that contract.

## Put behavior in its natural owner

- canonical ODE terms → `core/dynamics.py`;
- fixed-step arithmetic → `core/steppers.py`;
- trajectory orchestration → `core/solvers.py`;
- public boundary meanings → `core/boundary.py`;
- shared local aggregation → `backends/stencil.py`;
- high-level regular-grid behavior → `core/network.py`;
- optional domain I/O → domain modules, not the core;
- new public exports → deliberate package `__all__` surfaces plus reference coverage.

Avoid creating a generic plugin framework, backend hierarchy, registry, or configuration layer for a single concrete need when an existing owner can represent it directly.

## Add evidence at the same abstraction level

A formula change should have an analytic or differential oracle. A serialization change should have round-trip and compatibility evidence. A backend optimization should be compared against the semantic reference. A documentation-only scientific claim should either cite a verified source, derive the result, or execute a falsifiable example.

## Update documentation without duplicating truth

Docstrings own exact signatures and object behavior. Narrative pages own meaning, workflow, and scientific interpretation. When a type, default, or method list can be generated from the installed package, link to the API reference instead of copying it into several guides.

## Optional dependencies stay optional

Do not move Torch, CuPy, SciPy, DEAP, Pillow, Matplotlib, or documentation tooling into base runtime dependencies merely to simplify one implementation or example.

## Finish with artifact-level verification

Run the repository's standard lint/type/test lanes, build release artifacts where relevant, build documentation with strict warnings, and run the rendered API inventory checker. For optional behavior, use a lane containing the real dependency before claiming that behavior works.
