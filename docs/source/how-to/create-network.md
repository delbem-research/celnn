# Create a network

Use {py:class}`celnn.CellularNetwork` for the classical NumPy/CuPy reference simulator.

## Construct directly from coefficients

```python
import numpy as np
from celnn import CellularNetwork

u = np.zeros((32, 32), dtype=float)

network = CellularNetwork(
    input=u,
    initial_state=np.zeros_like(u),
    feedback=np.array(
        [[0.0, 0.1, 0.0],
         [0.1, 1.0, 0.1],
         [0.0, 0.1, 0.0]]
    ),
    control=np.zeros((3, 3)),
    bias=0.0,
    activation="piecewise_linear",
    boundary="reflect",
    device="cpu",
)
```

`input` determines the state shape. If `initial_state` is omitted, the network starts from zeros. Feedback and control templates are validated against the regular-grid topology.

## Construct from a reusable template

```python
from celnn import CellularNetwork
from celnn.templates import ONE_D_DIFFUSION

network = CellularNetwork.from_template(
    ONE_D_DIFFUSION,
    input=[0.0] * 64,
    boundary="reflect",
)
```

`from_template` validates the template, uses its optional initial state, copies its metadata, and lets explicit construction arguments such as activation, boundary, dtype, and device remain operational choices.

## Choose dtype deliberately

The classical network defaults to floating-point NumPy arrays with the library’s default dtype. Pass `dtype=` when numerical representation matters:

```python
network = CellularNetwork(input=u, dtype=np.float32)
```

Dtype is a numerical representation choice; it does not change the governing equation. For reproducible numerical comparisons, report it with the solver and tolerances.

## Reset state between experiments

```python
network.reset()
network.reset(initial_state=new_state)
```

`reset()` restores the stored initial state. Supplying a new state also makes that value the new reset point.

## Evaluate without running a trajectory

The public methods are useful for inspection:

```python
y = network.output(network.state)
drive = network.drive(network.state)
dxdt = network.derivative(network.state)
```

Use `derivative` when the question is about the continuous vector field rather than about one particular solver step.
