# celnn

`celnn` is a reusable Python package for **CelNN
(Cellular Neural Networks)** as locally connected nonlinear
dynamical systems over regular grids, signals, and image-like arrays.

Cellular Neural Networks are **not** Convolutional Neural Networks.
In this project, `CelNN` means a continuous-time cellular dynamical
system with local coupling, templates, states, outputs, inputs, and
bias terms.

## Installation

```bash
pip install celnn
```

For optional SciPy, image, and plotting support:

```bash
pip install celnn[all]
```

For GPU execution through CuPy/CUDA:

```bash
pip install celnn[gpu]
```

For genetic-algorithm-based template training (powered by DEAP):

```bash
pip install celnn[ga]
```

For differentiable templates and backpropagation through CelNN evolution:

```bash
pip install "celnn[torch]"
```

The same optional API includes modular Hebbian/Oja fast-weight plasticity with
explicit per-sequence state. See [the plasticity guide](docs/plasticity.md).

Use `device="gpu"` to require GPU execution, `device="auto"` to try GPU
and fall back to CPU, or `device="cpu"` for the default NumPy backend.

## Development and verification

Create the canonical development environment:

```bash
conda env create -f environment.yml
conda activate pycelnn
pip install -e . --no-deps
```

Run local checks:

```bash
python -m compileall src
pytest
ruff check .
mypy src/celnn/core/network.py src/celnn/core/simulation.py src/celnn/core/solvers.py
```

## Quick start

```python
import numpy as np
from celnn import CellularNetwork, SimulationConfig

u = np.random.rand(32, 32)

net = CellularNetwork(
    input=u,
    feedback=np.array([[0.0, 0.1, 0.0], [0.1, 1.0, 0.1], [0.0, 0.1, 0.0]]),
    control=np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]),
    bias=0.0,
    boundary="reflect",
    device="auto",
)

result = net.run(SimulationConfig(t_end=1.0, dt=0.01))
print(result.output.shape)
```

## Documentation shortcuts

* [CelNN: concept, theory, and study notes](docs/celnn.md)
* [Library usage and API guide](docs/libcelnn.md)
* [Examples](docs/examples.md)
* [Mathematical model](docs/mathematical-model.md)
* [Template design guide](docs/template-design.md)
* [Template creation guide](docs/template-creation-guide.md)
* [Migration from PyCNN](docs/migration-from-pycnn.md)
* [Differentiable network and PyTorch usage](docs/differentiable-network.md)

## Features

* Generic `CellularNetwork` API for 1D, 2D, and SciPy-backed ND simulations.
* Reusable `Template` and `TemplateRegistry` abstractions.
* Built-in activation functions, boundary modes, and solver options.
* Optional CuPy/CUDA backend for GPU local stencil aggregation.
* Optional image, signal, grid, serialization, and visualization helpers.
* Optional genetic-algorithm-based template trainer (DEAP).
* Optional `DifferentiableCellularNetwork` with learnable PyTorch templates.
* Demonstrative built-in templates for image processing, logic, diffusion, and pattern formation.
* Tests, examples, and technical documentation aimed at research and experimentation.

## Minimal example

```python
import numpy as np
from celnn import CellularNetwork, SimulationConfig
from celnn.activations import tanh_activation

signal = np.sin(np.linspace(0, 8 * np.pi, 512))

net = CellularNetwork(
    input=signal,
    feedback=np.array([0.2, 1.0, 0.2]),
    control=np.array([0.1, 0.8, 0.1]),
    bias=0.0,
    activation=tanh_activation,
    boundary="reflect",
)

result = net.run(SimulationConfig(t_end=5.0, dt=0.05))
print(result.output[:5])
```

## Implementation status

- Maturity: alpha (`0.1.x`), focused on regular-grid CelNN systems.
- CI-verified baseline: CPU/NumPy dynamics, SciPy solver path, templates,
  serialization, and image/signal utilities.
- GPU coverage:
  - deterministic CuPy backend behavior is tested in CI with a stubbed
    CuPy runtime.
  - real CUDA execution tests run when CuPy and a CUDA device are
    available (tests skip otherwise).

## License

This repository is distributed under the Apache-2.0 license. See [LICENSE](LICENSE).

## Attribution

`celnn` is an original, generalized library design inspired in part by
the MIT-licensed [PyCNN](https://github.com/ankitaggarwal011/PyCNN)
project, which focused on image processing with Cellular Neural
Networks.
