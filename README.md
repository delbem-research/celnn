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

Install only the optional capabilities you need:

```bash
pip install "celnn[scipy]"
pip install "celnn[image]"
pip install "celnn[viz]"
pip install "celnn[ga]"
pip install "celnn[torch]"
pip install "celnn[gpu]"
```

The `torch` API includes differentiable CelNN evolution and modular
Hebbian/Oja fast-weight plasticity with explicit per-sequence state.
See [the plasticity guide](docs/plasticity.md).

Use `device="gpu"` to require GPU execution, `device="auto"` to try GPU
and fall back to CPU, or `device="cpu"` for the default NumPy backend.

The classical NumPy/CuPy network supports `float32` and `float64`; the
default is `float64`. Native Euler solvers preserve the network dtype.
The optional SciPy `solve_ivp` path is intentionally `float64` only.

## Development and verification

Create the optional Conda development environment:

```bash
conda env create -f environment.yml
conda activate celnn
pip install -e . --no-deps
```

Run local checks:

```bash
python -m compileall src
pytest
ruff check .
mypy src/celnn
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
* Optional PyTorch plasticity and associative-memory APIs.
* Demonstrative built-in templates for image processing, logic, diffusion, and pattern formation.
* Tests, examples, and technical documentation aimed at research and experimentation.

## Persistence

The public JSON helpers in `celnn.io.serialization` write versioned,
validated artifacts atomically. New files use schema version 1. Loaders also
accept the current unversioned pre-0.4 representation as bounded legacy input.

Saved network artifacts preserve model meaning, including dtype, but do not
bind a model to the backend/device on which it happened to run. Loading
defaults to CPU unless another device is explicitly requested.

## Implementation status

- Maturity: alpha; public contracts may intentionally evolve before 1.0.
- CI verifies the base package on every advertised Python version, runs
  package-wide static checks, exercises representative optional integrations,
  and smoke-tests built wheel/sdist artifacts.
- GPU semantic parity is covered with deterministic backend tests; real CUDA
  claims require a CUDA-capable environment and are not inferred from stubs.

## License

This repository is distributed under the Apache-2.0 license. See [LICENSE](LICENSE).

## Attribution

`celnn` is an original, generalized library design inspired in part by
the MIT-licensed [PyCNN](https://github.com/ankitaggarwal011/PyCNN)
project, which focused on image processing with Cellular Neural
Networks.
