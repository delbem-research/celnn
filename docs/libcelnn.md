# celnn library usage and API guide

## Installation

Core install:

```bash
pip install celnn
```

Optional capabilities are explicit:

```bash
pip install "celnn[scipy]"
pip install "celnn[gpu]"
pip install "celnn[torch]"
pip install "celnn[image]"
pip install "celnn[viz]"
pip install "celnn[ga]"
```

The base package does not require those optional dependencies.

## Core execution model

`CellularNetwork` is the classical NumPy/CuPy API. Its dtype is part of the
numerical model representation:

- supported dtypes: `float32`, `float64`;
- default: `float64`;
- native `euler` and `semi_implicit_euler` preserve the network dtype;
- optional SciPy `solve_ivp` is intentionally `float64` only.

Backend choice changes execution, not the mathematical model. `device="cpu"`
uses NumPy, `device="gpu"`/`"cuda"` requires CuPy, and `device="auto"` may select
CuPy when available.

`DifferentiableCellularNetwork` is the optional PyTorch API. Its dtype/device
follow normal `torch.nn.Module` parameter/buffer ownership and PyTorch movement
semantics.

## Quick start

```python
import numpy as np
from celnn import CellularNetwork, SimulationConfig

u = np.random.uniform(-1.0, 1.0, size=(32, 32))

net = CellularNetwork(
    input=u,
    feedback=np.array(
        [[0.05, 0.2, 0.05], [0.2, 1.0, 0.2], [0.05, 0.2, 0.05]]
    ),
    control=np.zeros((3, 3)),
    bias=-0.1,
    boundary="wrap",
)

result = net.run(SimulationConfig(t_end=5.0, dt=0.05))
print(result.output.shape)
```

## `SimulationConfig`

```python
SimulationConfig(
    t_start=0.0,
    t_end=1.0,
    dt=0.01,
    solver="euler",
    return_trajectory=False,
    store_every=1,
    progress=False,
)
```

Supported solvers:

- `euler`;
- `semi_implicit_euler`;
- `solve_ivp` when SciPy is installed and the network dtype is `float64`.

CELNN does not currently expose a generic convergence certificate or numerical
stability-analysis API. Solver diagnostics must not be interpreted as such.

## `SimulationResult`

A result contains:

- final `state`;
- final `output`;
- stored `time`;
- optional `trajectory_state`;
- optional `trajectory_output`;
- execution metadata such as solver/backend/device.

## Templates and registries

`Template` bundles feedback/control kernels, bias, optional initial state,
description, tags, and metadata. `TemplateRegistry` owns reusable named
templates.

Both expose `to_dict()`/`from_dict()` for object-level representation. File
compatibility belongs to `celnn.io.serialization`.

## Persistence

Public JSON helpers:

- `save_config_json` / `load_config_json`;
- `save_template_json` / `load_template_json`;
- `save_registry_json` / `load_registry_json`;
- `save_network_json` / `load_network_json`.

New files use the envelope:

```json
{
  "schema_version": 1,
  "kind": "network",
  "data": {}
}
```

Loaders validate schema version, artifact kind, and payload. They also accept the
current unversioned pre-0.4 representation as explicit legacy-v0 input, but new
writes always use schema v1.

Network artifacts persist semantic state such as templates, state, activation,
boundary, dtype, and metadata. They do not persist backend/device as durable
identity. Loading defaults to CPU; request another device explicitly when
needed.

Writes are atomic.

## Optional APIs

- `celnn.training` requires the `ga` extra (DEAP).
- image helpers require the `image` extra (Pillow).
- visualization helpers require the `viz` extra (Matplotlib).
- `solve_ivp` requires the `scipy` extra.
- differentiable, plasticity, and associative-memory APIs require the `torch`
  extra.
- CuPy/CUDA execution requires the `gpu` extra.

Plain `import celnn` must remain valid without any of these extras installed.

## Boundary modes

- `constant`
- `wrap`
- `reflect`
- `nearest`
- `mirror`

Use `boundary_value` with `constant`.

## Extending CELNN

Keep new scientific behavior in its natural owner:

- ODE/dynamics in `celnn.core.dynamics`;
- time-integration formulas in `celnn.core.steppers`;
- shared stencil/boundary semantics in the existing backend boundary owners;
- domain-specific I/O outside the numerical core.

Do not introduce a new backend framework, plugin registry, or generic array
abstraction merely to add one implementation.

## Development verification

```bash
python -m compileall src
pytest
ruff check .
mypy src/celnn
```

CI additionally checks the advertised Python versions, representative optional
integrations, and built wheel/sdist artifacts in clean environments.
