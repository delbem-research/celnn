# Configure activation and boundaries

Activation and boundary handling both change the vector field. Treat them as model configuration, not cosmetic post-processing.

## Select a built-in activation

Pass a registered activation name:

```python
from celnn import CellularNetwork

network = CellularNetwork(
    input=u,
    feedback=A,
    control=B,
    activation="tanh",
)
```

The top-level package also exports the built-in activation functions directly when a callable is more convenient.

A custom callable can be used for execution, but `CellularNetwork.to_dict()` cannot automatically serialize arbitrary custom callables. Use a named built-in when built-in JSON persistence is required.

## Select a boundary mode

```python
network = CellularNetwork(
    input=u,
    feedback=A,
    control=B,
    boundary="constant",
    boundary_value=-1.0,
)
```

Supported modes are `constant`, `wrap`, `reflect`, `nearest`, and `mirror`. `boundary_value` is used only for a constant exterior.

Use `wrap` only for intentionally periodic domains. Use reflection or nearest-edge extension only when that continuation is meaningful for the experiment. For the operator-level consequences, see {doc}`../learn/boundaries-spatial-coupling`.

## Do not compensate for instability by changing boundaries blindly

If a trajectory is unstable, first determine whether the issue is the continuous model, the numerical method, or the spatial boundary. A boundary change alters the modeled operator; it is not a generic numerical stabilization parameter.
