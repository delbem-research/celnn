# Backends and semantic parity

Backends are execution representations of shared CELNN operations. They are not alternate scientific models.

## NumPy is the base reference path

The base package installs NumPy and exposes it through `NumPyBackend`. This path is available without optional execution dependencies.

## CuPy provides optional device execution

`CuPyBackend` uses CuPy/CUDA when explicitly requested or selected by `device="auto"`. Required GPU execution should fail if the capability is unavailable rather than silently converting a hardware claim into CPU execution.

## Torch supports differentiable array semantics

`TorchBackend` supplies the stencil primitives required by `DifferentiableCellularNetwork` while preserving tensors and autograd. It is exposed lazily so plain `import celnn` does not make PyTorch a base dependency.

## Parity means semantic agreement under a stated contract

Good backend parity tests hold model inputs, coefficients, boundary semantics, dtype, and numerical method constant and compare outputs within a justified tolerance.

For deterministic arithmetic with sufficiently high precision, tighter differential comparisons are preferable to broad tolerances.

## Hardware claims require hardware evidence

A CPU test of CuPy-facing code, a mock import, or structural inspection cannot establish CUDA performance or even successful execution on a real CUDA device. Documentation may state that a GPU backend exists and how it is selected; performance or hardware-specific claims require evidence from the relevant environment.

## Dtype is part of the comparison

`float32` and `float64` have different rounding behavior. A parity tolerance should therefore be selected with dtype and operation scale in view, not copied mechanically across all tests.
