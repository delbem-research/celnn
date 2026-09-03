# Scientific ownership and provenance

CELNN assigns each scientific or compatibility fact to a small number of explicit owners. Documentation explains those owners; it does not fork them into a second implementation.

| Concern | Definition / contract owner | Production owner | Primary evidence |
| --- | --- | --- | --- |
| Canonical CELNN vector field | normalized equation documented by the project | `src/celnn/core/dynamics.py` | analytic cases + dynamics tests |
| Fixed-step formulas | numerical method definition | `src/celnn/core/steppers.py` | exact formula/order/differential tests |
| Solver trajectory/result semantics | public simulation contract | `src/celnn/core/solvers.py` | solver tests + independent comparisons |
| Boundary semantics | public boundary names | `src/celnn/core/boundary.py` | boundary tests across implementations |
| Stencil aggregation | relative-offset stencil contract | `src/celnn/backends/stencil.py` | structural + backend parity tests |
| Reference network behavior | public API | `src/celnn/core/network.py` | network and serialization tests |
| Differentiable CELNN | top-level public API | `src/celnn/differentiable.py` | Torch/reference differential tests |
| Plastic fast weights | top-level public API | `src/celnn/plasticity.py` | rule/state tests |
| Associative memory | top-level public API | `src/celnn/associative.py`, `associative_field.py` | algebraic/state tests |
| GA template training | `celnn.training` public API | `src/celnn/training/` | deterministic seeded training tests |
| Exported API inventory | each module's `__all__` | public package modules | import tests + rendered `objects.inv` oracle |

## Literature provenance is separate from software ownership

Primary literature supports historical definitions, mechanisms, or prior algorithms. It does not become the source of truth for a software behavior unless CELNN explicitly adopts that behavior as its contract.

For example:

- Chua and Yang support the historical cellular architecture and classical local-dynamics model;
- Oja supports the lineage of a normalized Hebbian stabilizing term;
- Kozek, Roska, and Chua support genetic template-learning precedent;
- CELNN’s exact APIs, defaults, dtype rules, and state ownership come from the current package revision.

This distinction prevents a citation from being used to justify code behavior the paper never specified.

## The durable reasoning chain

Scientific changes should follow:

```text
definition → representation → invariant → oracle → evidence
```

A change is under-specified if any link is missing. A test that merely reproduces the implementation is weak evidence because it can repeat the same mistake.

## Documentation labs are evidence, not production owners

The {doc}`../labs/equilibrium` lab protects the published claim that a small Euler increment can coexist with a large vector-field residual. It intentionally does not require the current `approx_converged` heuristic to remain unchanged.

If a property is required of production CELNN behavior, regression ownership belongs in `tests/` as well as any explanatory lab.

## Version provenance

Implementation paths in these pages refer to the same repository revision used to build the site. When inspecting a versioned documentation build, use source from that revision rather than substituting a newer `main` branch.
