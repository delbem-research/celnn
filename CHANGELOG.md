# Changelog

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
