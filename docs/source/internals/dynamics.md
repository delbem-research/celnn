# Dynamics ownership

`src/celnn/core/dynamics.py` is the single production owner of the canonical continuous-time CELNN vector field.

It decomposes the computation into four functions:

```text
local_feedback(state)  → A * y(x)
local_control(input)   → B * u
local_drive(...)       → A * y(x) + B * u + z
derivative(...)        → -x + local_drive(...)
```

## Keep definition and execution separate

The dynamics functions depend on an `ArrayBackend` for local aggregation but do not choose NumPy, CuPy, or Torch semantics themselves. They receive the backend and boundary configuration from the caller.

This makes the equation independent of execution representation.

## `CellularNetwork` delegates rather than redefines

The public `CellularNetwork.drive()` and `CellularNetwork.derivative()` methods coerce user state to the network dtype and delegate to these canonical functions. They are inspection surfaces, not second definitions of the equation.

`DifferentiableCellularNetwork` also delegates to the same dynamics helpers using the Torch backend. This is why parity between reference and differentiable execution is structurally meaningful.

## Extension rule

A change to the actual CELNN differential equation belongs here first, with an explicit mathematical definition and production tests. A new high-level class should not copy the derivative and diverge silently.

If a new effect is external to the canonical CELNN equation—for example caller-owned cross-channel mixing—prefer explicit composition rather than broadening the core equation without a deliberate scientific contract.
