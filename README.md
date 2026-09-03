# celnn

`celnn` is a Python library that makes **Cellular Neural Networks (CelNNs)** practical and reusable for scientific and engineering applications, providing a general computational framework for continuous-time, locally coupled nonlinear dynamical systems over regular grids, signals, and image-like arrays.

In this project, **CelNN** refers to this cellular dynamical-system model, **not** to Convolutional Neural Networks.

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

The existing compatibility bundle also remains available:

```bash
pip install "celnn[all]"
```

The `all` extra preserves the pre-existing SciPy/GPU/image/viz/GA bundle;
PyTorch remains an explicit `torch` capability.

The `torch` API includes differentiable CelNN evolution and modular
Hebbian/Oja fast-weight plasticity with explicit per-sequence state.
See [the plasticity guide](docs/source/how-to/plasticity.md).

Use `device="gpu"` to require GPU execution, `device="auto"` to try GPU
and fall back to CPU, or `device="cpu"` for the default NumPy backend.

The classical network defaults to `float64`. Explicit dtype behavior remains
compatibility-sensitive; native `float32` and `float64` execution paths are
covered directly by the test suite. This does not impose a new
`float64`-only restriction on the optional SciPy `solve_ivp` path.

## Development

For the canonical development setup and local verification workflow, see
[CONTRIBUTING.md](CONTRIBUTING.md). The repository continues to provide its
existing Conda development environment while Pyright is the maintained static
type checker.

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
print(result.convergence)
```

`SimulationConfig.stability_checks` and `SimulationResult.convergence` remain
part of the public interface. The convergence mapping is a numerical diagnostic,
not a formal equilibrium certificate; see the documentation for the scientific
interpretation.

## Documentation

The technical knowledge system is organized by reader intent:

* [Start here: what is a Cellular Neural Network?](docs/source/start/what-is-celnn.md)
* [Learn the model from first principles](docs/source/learn/equation.md)
* [Use CELNN: task-oriented guides](docs/source/how-to/index.md)
* [Scientific explanations](docs/source/explanation/dynamical-systems.md)
* [Generated public API reference](docs/source/reference/index.md)
* [Internals and verification model](docs/source/internals/architecture.md)
* [Scientific bibliography](docs/source/bibliography.md)

The documentation is built with Sphinx/MyST and published through the repository's GitHub Pages workflow. Source links above remain useful when browsing a branch or pull request before publication.

## Features

* Generic `CellularNetwork` API for regular-array simulations.
* Reusable `Template` and `TemplateRegistry` abstractions.
* Built-in activation functions, boundary modes, and solver options.
* Optional SciPy ODE integration.
* Optional CuPy/CUDA backend for GPU local stencil aggregation.
* Optional image, signal, grid, serialization, and visualization helpers.
* Optional genetic-algorithm-based template trainer (DEAP).
* Optional `DifferentiableCellularNetwork` with learnable PyTorch templates.
* Optional PyTorch plasticity and associative-memory APIs.
* Demonstrative built-in templates for image processing, logic, diffusion, and pattern formation.
* Tests, executable documentation, and generated API reference aimed at research and engineering use.

## Persistence

The public JSON helpers in `celnn.io.serialization` preserve the established flat
JSON representation and write files atomically. Loading remains compatible with
previously accepted payloads rather than introducing a new schema envelope or
stricter migration contract.

Saved network artifacts preserve model meaning, including dtype, but new writes
do not bind the model to the backend/device on which it happened to run. Loaders
continue accepting historical payloads containing those operational fields and
loading defaults to CPU unless another device is explicitly requested.

## Implementation status

- Release version and maturity are defined by the package metadata; compatibility
  changes should be explicit rather than incidental.
- CI verifies the base package on every advertised Python version, runs
  package-wide Ruff/Pyright checks, exercises representative optional
  integrations and dependency floors, and smoke-tests built wheel/sdist
  artifacts.
- The installed wheel ships `py.typed` and its public typing surface is verified
  from the built artifact.
- GPU semantic parity is covered with deterministic backend tests; real CUDA
  claims require a CUDA-capable environment and are not inferred from stubs.

## License

This repository is distributed under the Apache-2.0 license. See [LICENSE](LICENSE).

## Attribution

`celnn` is an original, generalized library design inspired in part by
the MIT-licensed [PyCNN](https://github.com/ankitaggarwal011/PyCNN)
project, which focused on image processing with Cellular Neural
Networks.
