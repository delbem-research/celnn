# Work with data and persist artifacts

Keep domain preparation and persistence outside the numerical core. The core network consumes arrays; helper modules convert or generate domain-specific data around it.

## Generate simple grids and signals

`celnn.domains` exposes deterministic grid constructors and signal helpers. For stochastic grid or noisy-signal generation, pass or document the relevant seed/randomness in scientific examples so results remain reproducible.

```python
from celnn.domains import checkerboard_grid, impulse_grid

u = impulse_grid((64, 64))
mask = checkerboard_grid((64, 64))
```

See the generated Domain helpers reference for exact signatures.

## Load and save grayscale images

Install Pillow support:

```bash
python -m pip install "celnn[image]"
```

Then use the image-specific module explicitly:

```python
from celnn.domains.image import load_grayscale, save_grayscale

u = load_grayscale("input.png")
result = network.run(config)
save_grayscale(result.output, "output.png")
```

The image helper normalizes ordinary grayscale image data to the package’s `[-1, 1]` convention and maps normalized output back to `uint8` for saving. Image I/O is not part of the CELNN differential equation.

## End-to-end demonstrative image run

The built-in `EDGE_DETECTION` template is explicitly demonstrative and PyCNN-inspired; using it here shows the data flow without claiming that its coefficients are optimal for a particular image or metric.

```python
from celnn import CellularNetwork, SimulationConfig
from celnn.domains.image import load_grayscale, save_grayscale
from celnn.templates import EDGE_DETECTION

u = load_grayscale("input.png")

network = CellularNetwork.from_template(
    template=EDGE_DETECTION,
    input=u,
    activation="piecewise_linear",
    boundary="reflect",
)
result = network.run(SimulationConfig(t_end=5.0, dt=0.05))
save_grayscale(result.output, "edge.png")
```

For a real application, define the property that matters—such as localization, noise sensitivity, or agreement with a reference result—and validate that property instead of treating a demonstrative built-in as a benchmark.

## Persist templates and configurations

```python
from celnn.io import save_config_json, save_template_json

save_template_json(template, "template.json")
save_config_json(config, "simulation.json")
```

Corresponding `load_*` functions restore each object.

## Persist a network

```python
from celnn.io import save_network_json, load_network_json

save_network_json(network, "network.json")
restored = load_network_json("network.json")
```

The serialized network represents semantic model state such as input, templates, activation, boundary, dtype, metadata, initial state, and current state. New writes do not make the transient execution backend/device durable model identity; loading defaults to CPU unless another device is requested explicitly.

JSON writes are atomic: the helper writes a temporary file in the target directory, flushes it, then replaces the destination.
