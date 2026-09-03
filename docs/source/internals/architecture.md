# Architecture and dependency boundaries

CELNN keeps the scientific core small and pushes optional capabilities to explicit boundaries. The intended dependency direction is:

```text
public API
    ↓
core model and validation
    ↓
canonical dynamics + time steppers
    ↓
shared stencil semantics
    ↓
backend-specific array primitives
```

Optional domain, training, PyTorch, visualization, image, SciPy, and GPU capabilities sit around this path rather than redefining it.

## Core public path

{py:class}`celnn.CellularNetwork` is the reference high-level owner. It validates the regular-grid model, resolves the activation and backend, and delegates scientific computation instead of embedding alternative copies of it.

`src/celnn/core/dynamics.py` owns the canonical decomposition of feedback, control, drive, and derivative. `src/celnn/core/steppers.py` owns the fixed-step update formulas. `src/celnn/core/solvers.py` orchestrates trajectories and result construction.

## Spatial execution path

`src/celnn/backends/stencil.py` owns the generic pad-and-shift stencil algorithm. NumPy, CuPy, and Torch backends provide array-specific primitives or validated fast paths. This is the mechanism that prevents each backend from inventing a slightly different neighborhood computation.

Boundary terminology is normalized in `src/celnn/core/boundary.py` before backend libraries apply their own naming conventions.

## Optional capability boundaries

- SciPy is imported only when `solve_ivp` is selected.
- CuPy is an optional execution backend.
- PyTorch is loaded lazily for differentiable, plasticity, associative-memory, or Torch-backend APIs.
- DEAP is imported when GA execution is requested.
- Pillow and Matplotlib remain domain/visualization helpers, not core requirements.

A plain base installation therefore remains NumPy-only.

## Why this separation matters scientifically

When multiple execution paths reuse one dynamics definition and one step formula, differential tests compare representations of the same intended model rather than two independently coded equations. That reduces semantic drift and makes a failing parity test easier to localize.

Do not add an abstraction layer merely because another package could theoretically be supported. Add a boundary only when a concrete capability requires one and the existing owner cannot represent it cleanly.
