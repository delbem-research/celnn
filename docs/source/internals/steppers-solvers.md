# Steppers and solvers

CELNN deliberately separates **one-step numerical formulas** from **trajectory orchestration**.

## Steppers own arithmetic

`src/celnn/core/steppers.py` is the only owner of fixed-step update formulas used by both the NumPy reference simulator and differentiable PyTorch execution.

The functions are pure arithmetic over array-like values:

```text
euler_step(state, dt, derivative)
semi_implicit_euler_step(state, dt, drive)
```

Keeping these formulas array-agnostic lets gradients pass through Torch tensors without creating a second numerical definition.

## Solvers own trajectory mechanics

`src/celnn/core/solvers.py` owns:

- selecting Euler, semi-implicit Euler, or SciPy `solve_ivp`;
- iterating requested time points;
- trajectory storage and `store_every` semantics;
- updating the network's final state;
- constructing `SimulationResult`;
- recording solver metadata and the current state-delta diagnostic.

The solver does not own the CELNN derivative itself.

## `solve_ivp` is intentionally separate

The SciPy path flattens state for the external integrator and reshapes it inside the RHS callback before delegating to `network.derivative`. This gives CELNN an independent integration implementation without duplicating the vector field.

## Convergence metadata is not a definition

The current `_convergence_info` function measures only the maximum absolute difference between the final two solver states and compares it with a fixed threshold. This is a compatibility diagnostic in the solver owner, not the definition of equilibrium.

A future redesign may strengthen that diagnostic without changing the mathematical meaning of equilibrium or invalidating the documentation lab.
