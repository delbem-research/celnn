# Stencils, topology, and boundaries

Spatial coupling is distributed across three owners with distinct responsibilities.

## Topology owns geometric validity

`src/celnn/core/topology.py` represents the regular-grid shape and validates template geometry. The network uses that topology to establish which array axes are spatial and which template extents are admissible.

## Boundary utilities own public edge semantics

`src/celnn/core/boundary.py` validates the public modes `constant`, `wrap`, `reflect`, `nearest`, and `mirror` and maps those meanings to library-specific pad names.

NumPy and SciPy invert the names used for two reflection conventions. The mapping code exists so that CELNN’s public names retain one meaning across backends.

## `StencilBackend` owns aggregation

`src/celnn/backends/stencil.py` implements the shared algorithm:

1. prepare field and kernel arrays;
2. determine the spatial axes;
3. use a validated backend fast path when one exists, otherwise pad the field;
4. iterate over kernel offsets;
5. multiply the aligned shifted window by its coefficient;
6. accumulate the result.

Backend subclasses supply padding, allocation, conversion, and optional acceleration primitives.

## Why the shared loop is important

If every backend implemented its own stencil interpretation, a parity test would compare separate algorithms with separate opportunities for offset or boundary mistakes. A shared semantic loop concentrates that risk in one owner and makes backend differences mostly representation differences.

## Fast paths need differential evidence

An accelerated path is correct only if it preserves the shared stencil contract. Tests should compare it against the independent structural implementation over representative shapes, kernels, dtypes, and boundary modes.

Performance alone is not evidence of semantic equivalence.
