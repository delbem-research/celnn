# Changelog

## Unreleased

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
