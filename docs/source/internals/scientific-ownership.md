# Scientific ownership and evidence

CELNN keeps scientific meaning in a small number of explicit owners. The
documentation interprets those owners; it must not fork their semantics.

| Concern | Production owner | Evidence role |
| --- | --- | --- |
| CelNN differential equation | `src/celnn/core/dynamics.py` | analytical/differential tests |
| Time-integration formulas | `src/celnn/core/steppers.py` | analytical/order tests |
| Solver orchestration and current convergence metadata | `src/celnn/core/solvers.py` | solver tests and scientific falsifiers |
| Stencil semantics | `src/celnn/backends/stencil.py` | backend differential tests |
| Public top-level surface | `src/celnn/__init__.py::__all__` | API/import tests + generated reference |

The durable maintainer contract is `AGENTS.md` in the same repository revision.
It requires scientific changes to follow definition → representation → invariant
→ oracle → evidence.

## Change rule

A documentation lab can verify that a published example still supports its
claim. It is not the sole owner of production correctness. If a property is a
required CELNN invariant, its regression protection belongs in `tests/` as well.

For example, the [equilibrium lab](../labs/equilibrium.md) demonstrates that a
small Euler increment does not imply a small vector-field residual. It does not
turn today's `approx_converged` implementation into the mathematical definition
of equilibrium.

## Version provenance

The production-owner paths above are identifiers in the same Git revision used
to build this documentation. Inspect those version-local paths when reasoning
about implementation details; do not substitute the current `main` branch for a
versioned documentation build.
