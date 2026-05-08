# celnn library usage and API guide

## Installation

Core install:

```bash
pip install celnn
```

Optional extras:

```bash
pip install celnn[scipy]
pip install celnn[image]
pip install celnn[viz]
pip install celnn[all]
```

## Package structure

```text
celnn/
  core/            numerical engine and core abstractions
  backends/        backend protocol and NumPy implementation
  domains/         image, signal, and grid utilities
  templates/       built-in templates and registry
  io/              JSON serialization helpers
  visualization/   optional plotting helpers
  utils/           small shared utilities
```

## Quick start

```python
import numpy as np
from celnn import CellularNetwork, SimulationConfig

u = np.random.uniform(-1.0, 1.0, size=(32, 32))

net = CellularNetwork(
    input=u,
    feedback=np.array([[0.05, 0.2, 0.05], [0.2, 1.0, 0.2], [0.05, 0.2, 0.05]]),
    control=np.zeros((3, 3)),
    bias=-0.1,
    boundary="wrap",
)

result = net.run(SimulationConfig(t_end=5.0, dt=0.05))
print(result.output.shape)
```

## Core classes

### `CellularNetwork`

Main user-facing object. It stores:

- input field,
- current internal state,
- feedback and control templates,
- bias,
- activation,
- boundary behavior,
- topology metadata,
- runtime metadata.

Constructor:

```python
CellularNetwork(
    input,
    state_shape=None,
    initial_state=None,
    feedback=None,
    control=None,
    bias=0.0,
    activation="piecewise_linear",
    boundary="constant",
    boundary_value=0.0,
    dtype=None,
    metadata=None,
)
```

Default behavior:

- `state_shape` defaults to `input.shape`.
- `initial_state` defaults to zeros.
- `feedback` defaults to a centered identity-like stencil.
- `control` defaults to a zero stencil with the same shape as `feedback`.
- `bias` can be scalar or broadcastable array.

Key methods:

- `run(config=None) -> SimulationResult`
- `derivative(state) -> np.ndarray`
- `output(state=None) -> np.ndarray`
- `step(dt) -> np.ndarray`
- `reset(initial_state=None) -> None`
- `from_template(...)`
- `to_dict()`
- `from_dict()`

### `Template`

Reusable template object bundling:

- `name`
- `feedback`
- `control`
- `bias`
- optional `initial_state`
- `description`
- `tags`
- `metadata`

Key methods:

- `validate()`
- `to_dict()`
- `from_dict()`
- `copy()`
- `with_bias()`
- `as_arrays()`

### `TemplateRegistry`

Registry for named templates.

Methods:

- `register(template, overwrite=False)`
- `get(name)`
- `list()`
- `names()`
- `remove(name)`
- `to_dict()`
- `from_dict()`

### `SimulationConfig`

Controls time integration:

```python
SimulationConfig(
    t_start=0.0,
    t_end=1.0,
    dt=0.01,
    solver="euler",
    return_trajectory=False,
    store_every=1,
    dtype=None,
    stability_checks=True,
    progress=False,
)
```

Supported solver names:

- `euler`
- `semi_implicit_euler`
- `solve_ivp` if SciPy is installed

### `SimulationResult`

Holds:

- final `state`
- final `output`
- stored `time`
- optional `trajectory_state`
- optional `trajectory_output`
- `metadata`
- optional `convergence`

## Activation functions

Built-ins:

- `piecewise_linear`
- `saturated_linear`
- `identity`
- `tanh_activation`
- `sigmoid_activation`
- `sign_activation`
- `relu_activation`

You may also pass a callable directly. Named built-ins serialize automatically. Arbitrary callables do not.

## Solvers

### Explicit Euler

Simple, transparent, and good for experiments. It is the default.

### Semi-implicit Euler

Treats the linear decay `-x` implicitly while keeping the nonlinear local drive explicit. This often behaves better than plain Euler for the same timestep.

### SciPy `solve_ivp`

Available when SciPy is installed. Useful for comparison runs and higher-dimensional aggregation because the SciPy backend also supports ND stencils more naturally.

## Boundary modes

- `constant`
- `wrap`
- `reflect`
- `nearest`
- `mirror`

Use `boundary_value` with `constant`.

## Image domain utilities

`celnn.domains.image` is optional and depends on Pillow.

Functions:

- `load_grayscale`
- `save_grayscale`
- `normalize_image`
- `denormalize_image`
- `image_to_array`
- `array_to_image`

The core package never imports Pillow directly.

## Signal domain utilities

`celnn.domains.signal` provides:

- `normalize_signal`
- `generate_sine_wave`
- `generate_noisy_sine`
- optional `plot_signal`
- optional `plot_signal_comparison`

## Grid domain utilities

`celnn.domains.grid` provides:

- `random_grid`
- `impulse_grid`
- `checkerboard_grid`
- `coordinate_grid`

## Serialization

JSON helpers live in `celnn.io.serialization`.

Available helpers:

- `save_config_json`
- `load_config_json`
- `save_template_json`
- `load_template_json`
- `save_registry_json`
- `load_registry_json`
- `save_network_json`
- `load_network_json`

## Visualization

`celnn.visualization.plots` provides optional Matplotlib wrappers:

- `plot_signal`
- `plot_grid`
- `plot_trajectory`

## Error handling

Custom exceptions:

- `CelNNError`
- `ShapeMismatchError`
- `TemplateValidationError`
- `SolverError`
- `BackendError`
- `OptionalDependencyError`

## Extending the library

### Adding a new template

Create a `Template` and register it:

```python
from celnn.templates import Template, TemplateRegistry

template = Template(
    name="my_template",
    feedback=[[0.0, 0.2, 0.0], [0.2, 1.0, 0.2], [0.0, 0.2, 0.0]],
    control=[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
    bias=0.0,
)

registry = TemplateRegistry()
registry.register(template)
```

### Adding a new activation function

Pass a callable:

```python
import numpy as np

def softsign(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.abs(x))
```

### Adding a new domain adapter

Keep domain-specific I/O and preprocessing outside `celnn.core`. Domain adapters should convert to and from NumPy arrays without introducing hard dependencies into the numerical core.

### Future backend design

The backend protocol is intentionally small. Version 0.1 includes a
NumPy backend and an optional CuPy/CUDA backend. A future JAX, PyTorch,
or Numba backend would mainly need:

- array conversion,
- local stencil aggregation,
- compatible activation handling.

### GPU execution

Install GPU support with:

```bash
pip install celnn[gpu]
```

Then select the device on `CellularNetwork`:

```python
net = CellularNetwork(
    input=u,
    feedback=A,
    control=B,
    boundary="reflect",
    device="gpu",
)
```

Device options:

- `cpu`: use the NumPy backend. This is the default.
- `gpu` or `cuda`: require the CuPy/CUDA backend and fail clearly if it
  cannot run.
- `auto`: use CuPy/CUDA when available, otherwise fall back to NumPy.

The public result arrays are returned as NumPy arrays. The CuPy backend
accelerates local stencil aggregation on the GPU.

## Complete examples

See [docs/examples.md](examples.md) and the `examples/` directory for:

- image edge detection,
- 1D signal filtering,
- custom template simulation,
- reaction-diffusion-like pattern experiments,
- custom activations,
- template registry usage.

For a step-by-step guide on creating new templates for new tasks, see
[docs/template-creation-guide.md](template-creation-guide.md).

## FAQ

### Does this package implement Convolutional Neural Networks?

No. `celnn` implements CelNN systems, not Convolutional Neural
Networks.

### Does v0.1 support arbitrary graph neighborhoods?

Not yet. The public API is designed with extension points, but v0.1 focuses on regular arrays.

### Does v0.1 support ND arrays?

Yes at the API level. Robust 1D and 2D support is available with the NumPy fallback. ND aggregation is best used with SciPy installed.

### Why keep image helpers outside the core?

Because the mathematical core is generic and should not depend on optional image libraries.
