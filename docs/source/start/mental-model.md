# Mental model

The shortest correct mental model of a CELNN simulation is:

```text
external input u ── B ──┐
                        │
state x ── y(x) ── A ───┼──> local drive ──> dx/dt ──> integrator ──> next x
                        │
                 bias z ┘
```

The feedback path makes the system recurrent: the current state produces an output, neighboring outputs influence the derivative, and integration produces the state used at the next instant.

## Five objects to keep separate

**State `x`** is the dynamical memory of the system. It is what the ODE integrates.

**Output `y(x)`** is the observable field after the selected activation/output function. Feedback acts on this output, not directly on the raw state in the canonical equation.

**Input `u`** is the external field. In the reference simulator it is fixed for a network instance; it is not the same thing as the state or the initial condition.

**Feedback `A`** aggregates neighboring outputs. It determines recurrent spatial coupling.

**Control `B`** aggregates neighboring external inputs. It determines how the fixed input drives each cell.

The bias `z` adds a local offset to the drive.

## The software mapping

In CELNN, {py:class}`celnn.CellularNetwork` owns the input, current state, activation, templates, boundaries, and execution backend. Its public `derivative(state)` method evaluates the canonical vector field for a supplied state. {py:class}`celnn.SimulationConfig` describes how that vector field is integrated over a time interval, and {py:class}`celnn.SimulationResult` holds the final fields plus optional stored trajectories and diagnostics.

This separation is useful when reasoning about failures. If a spatial response is wrong, inspect templates and boundary semantics. If a trajectory is inaccurate or unstable, inspect the numerical method and step size. If a reported convergence diagnostic is misleading, inspect the diagnostic itself rather than changing the underlying ODE.

## Three layers of meaning

A CELNN experiment always contains three layers:

1. **model** — the vector field defined by `A`, `B`, `z`, activation, input, and boundary semantics;
2. **numerical representation** — arrays, dtype, backend, and integration method;
3. **evidence** — observations or assertions used to justify a claim about the run.

Keeping these layers distinct prevents a numerical artifact from being mistaken for a property of the mathematical system.

Continue with {doc}`first-system` for a minimal executable example.
