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

The durable maintainer contract is in
[`AGENTS.md`](https://github.com/delbem-research/celnn/blob/main/AGENTS.md).
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

## Source links

- [core/dynamics.py](https://github.com/delbem-research/celnn/blob/main/src/celnn/core/dynamics.py)
- [core/steppers.py](https://github.com/delbem-research/celnn/blob/main/src/celnn/core/steppers.py)
- [core/solvers.py](https://github.com/delbem-research/celnn/blob/main/src/celnn/core/solvers.py)
- [backends/stencil.py](https://github.com/delbem-research/celnn/blob/main/src/celnn/backends/stencil.py)
- [tests/](https://github.com/delbem-research/celnn/tree/main/tests)
