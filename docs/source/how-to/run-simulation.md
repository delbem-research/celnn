# Run a simulation

Create a {py:class}`celnn.SimulationConfig` and pass it to `CellularNetwork.run()`.

```python
from celnn import SimulationConfig

config = SimulationConfig(
    t_start=0.0,
    t_end=5.0,
    dt=0.01,
    solver="euler",
    return_trajectory=True,
    store_every=10,
    stability_checks=True,
    progress=False,
)

result = network.run(config)
```

## Choose the time interval

`t_start` and `t_end` define the requested interval. `dt` must be positive. If the interval is not an exact multiple of `dt`, the configuration includes the final time explicitly rather than silently ending early.

## Choose a solver

- `euler` — explicit Euler; no optional dependency;
- `semi_implicit_euler` — treats the linear leak term implicitly;
- `solve_ivp` — requires the `scipy` extra.

See {doc}`../learn/time-evolution` before interpreting solver differences scientifically.

## Store only the trajectory resolution you need

Set `return_trajectory=True` to store state/output trajectories. `store_every` controls retention frequency for the fixed-step solvers; it does not reduce the number of integration steps.

For long runs where only the endpoint is required, leave `return_trajectory=False` to avoid unnecessary trajectory storage.

## Treat `stability_checks` as diagnostics

The current network may add a warning to result metadata for obviously large explicit steps. This is not a general stability proof. Required numerical validity should be established by problem-specific analysis or convergence evidence.
