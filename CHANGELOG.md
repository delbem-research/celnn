# Changelog

## Unreleased

### Changed

- Consolidated numerical contracts for the next minor release while preserving
  established compatibility: `float64` remains the default dtype, native
  `float32`/`float64` execution is explicitly verified, and no new dtype or
  `solve_ivp` exclusion is imposed by this consolidation.
- Preserved the existing `convergence` result field and `stability_checks`
  configuration surface. Scientific redesign of either diagnostic is deferred
  to a dedicated change rather than being mixed into this consolidation.
- Preserved the established flat JSON representation and permissive loading
  behavior while making durable writes atomic.
- Removed backend/device identity from newly persisted network truth while
  continuing to accept historical payloads that contain those operational
  fields; dtype remains semantic model state.
- Preserved the existing `all` compatibility extra and Conda development
  environment instead of changing packaging/development interfaces here.
- Replaced Mypy with Pyright as the maintained package-wide static type checker.
- Made `pyproject.toml` the release-version owner and derive
  `celnn.__version__` from installed package metadata.
- Expanded the verification contract to the supported Python matrix,
  package-wide Pyright, optional-dependency isolation, dependency floors, and
  installed wheel/sdist smoke tests.
- Normalized project metadata and documentation to
  `https://github.com/delbem-research/celnn`.

### Added

- Added `AGENTS.md` with the repository's durable scientific and engineering
  maintainer contract.
- Added `py.typed` and installed-artifact type verification for the public
  `celnn` API.

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
