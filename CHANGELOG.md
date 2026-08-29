# Changelog

## Unreleased

### Changed

- Consolidated scientific contracts for the next minor release: classical
  networks support `float32`/`float64`, native solvers preserve the selected
  dtype, and SciPy `solve_ivp` is explicitly `float64` only.
- Removed the previous heuristic convergence and generic timestep-stability
  claims instead of preserving scientifically ambiguous API.
- Versioned public JSON artifacts with schema v1, bounded legacy-v0 reading,
  strict validation, CPU-portable network loading, and atomic writes.
- Removed persisted backend/device identity from network artifacts; dtype
  remains part of the semantic representation.
- Removed the umbrella `all` extra in favor of explicit capability extras.
- Made `pyproject.toml` the release-version owner and derive
  `celnn.__version__` from installed package metadata.
- Expanded the verification contract to the supported Python matrix,
  package-wide Mypy, optional-dependency isolation, and installed wheel/sdist
  smoke tests.
- Normalized project metadata and documentation to
  `https://github.com/delbem-research/celnn`.

### Added

- Added `AGENTS.md` with the repository's durable scientific and engineering
  maintainer contract.
- Added `py.typed` and a contract test for the deliberate top-level
  `celnn.__all__` API.

### Existing unreleased work

- Added modular PyTorch fast-weight plasticity with explicit per-sequence
  `PlasticityState`, reusable `HebbianRule` and `OjaRule` updates, configurable
  slow/fast-weight composition, and a functional `PlasticLinear` layer.
- Added the optional `torch` extra and the top-level
  `celnn.DifferentiableCellularNetwork` API for learnable or frozen templates,
  multistep backpropagation, diagonal multichannel states, and conversion to
  and from classical one-dimensional `Template` objects.
- Reused the canonical CelNN dynamics, activation functions, stencil
  aggregation, boundary semantics, and integration formulas across NumPy,
  CuPy, and PyTorch execution paths.

## 0.1.0 - 2026-04-24

- Initial `celnn` package scaffold.
- Added a generic `CellularNetwork` API for 1D, 2D, and SciPy-backed ND simulations.
- Added reusable templates, registry, serialization, domain utilities, examples, and tests.
- Added technical documentation covering theory, API usage, templates, examples, and migration from PyCNN.
- Added an optional genetic-algorithm-based template trainer in `celnn.training`,
  backed by [DEAP](https://deap.readthedocs.io/en/master/) and exposed as the
  `ga` extra (`pip install celnn[ga]`).
- Added a CuPy/CUDA backend for GPU local stencil aggregation, exposed as the
  `gpu` extra (`pip install celnn[gpu]`).
