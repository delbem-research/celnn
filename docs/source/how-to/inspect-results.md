# Inspect results and trajectories

{py:class}`celnn.SimulationResult` contains the simulation output without hiding how much trajectory information was retained.

```python
result = network.run(config)

final_state = result.state
final_output = result.output
final_time = result.final_time
```

## State and output

`state` is the final internal state. `output` is the activation applied to that state. They can be numerically identical for identity activation but should still be interpreted as different model quantities.

## Trajectory data

```python
if result.has_trajectory:
    states = result.trajectory_state
    outputs = result.trajectory_output
    times = result.time
```

When no trajectory was requested, `time` contains the final stored time and the trajectory arrays are `None`.

## Execution metadata

The solver records metadata including solver, boundary, backend, device, and shape; warnings may also be present. Use metadata when comparing runs so that a backend or solver change is not mistaken for a change in the model definition.

## Convergence diagnostics

`result.convergence` currently reports a last-step state-difference diagnostic. Do not interpret `approx_converged` as proof that the final vector-field residual is small. For equilibrium studies, evaluate the derivative explicitly:

```python
residual = network.derivative(result.state)
```

Then report both the residual norm and the state-increment diagnostic if both are relevant. See {doc}`../learn/equilibrium-convergence-stability`.
