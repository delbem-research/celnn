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

The existing compatibility bundle also remains available:

```bash
pip install "celnn[all]"
```

The base package does not require optional capabilities. The `all` extra keeps
the established SciPy/GPU/image/viz/GA bundle; PyTorch remains an explicit
`torch` capability.

## Core execution model

`CellularNetwork` is the classical NumPy/CuPy API. Its dtype is part of the
numerical representation. The default is `float64`, and explicitly selected
dtype behavior is compatibility-sensitive rather than being narrowed by this
consolidation. Native `float32` and `float64` execution paths are verified
explicitly.

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
print(result.convergence)
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
    stability_checks=True,
    progress=False,
)
```

Supported solvers:

- `euler`;
- `semi_implicit_euler`;
- `solve_ivp` when SciPy is installed.

This consolidation does not add a new dtype-only restriction to `solve_ivp`.
`stability_checks` remains part of the established configuration interface.
Its scientific redesign, if needed, belongs in a dedicated change.

## `SimulationResult`

A result contains:

- final `state`;
- final `output`;
- stored `time`;
- optional `trajectory_state`;
- optional `trajectory_output`;
- execution metadata such as solver/backend/device;
- the established `convergence` diagnostic mapping.

`convergence` is preserved for compatibility. It should not be treated as a
formal mathematical convergence certificate unless and until a dedicated
scientific contract defines such semantics.

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

The public helpers preserve the established flat JSON representation. This
consolidation does not introduce a schema envelope, migration framework, or
stricter deserialization contract because external artifacts may exist outside
the repository.

Network artifacts persist semantic state such as templates, state, activation,
boundary, dtype, and metadata. New writes do not persist backend/device as
durable model identity, while loaders continue accepting historical payloads
that contain those operational fields. New artifacts default to CPU; historical
artifacts continue selecting a stored device for compatibility. Pass
`device="cpu"` explicitly when loading an untrusted legacy artifact.

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

For the canonical development setup and local verification workflow, see
[CONTRIBUTING.md](../CONTRIBUTING.md). The existing Conda environment remains
supported, and Pyright is the maintained package-wide static type checker.
