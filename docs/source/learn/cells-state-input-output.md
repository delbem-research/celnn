# Cells, state, input, and output

The original Cellular Neural Network architecture is built from repeated cells connected through local neighborhoods. Chua and Yang define the cell’s input, state, and output as distinct circuit variables and use local controlled sources to couple neighboring inputs and outputs. See {ref}`chua-yang-1988-theory`.

The CELNN software model preserves the same conceptual separation without reproducing the circuit implementation.

## State is what evolves

Let `x(t)` denote the array of all cell states at time `t`. The state is the dependent variable of the ODE:

$$
\dot{x}(t) = f(x(t); u, A, B, z).
$$

An **initial state** is only the value `x(0)`. It is not a second input channel. Changing the initial state can change a transient or select a different basin of attraction even when `u`, `A`, `B`, and `z` are unchanged.

In {py:class}`celnn.CellularNetwork`, `state` is mutable simulation state and `reset()` restores or replaces the initial state.

## Input is external drive

The external input `u` participates in the vector field through the control template `B`. The classical papers often normalize circuit input magnitudes to a prescribed range; that is a property of the circuit model studied there, not a universal array-range restriction imposed by this Python package.

The reference `CellularNetwork` stores one input array for the lifetime of the network object. Time-varying forcing is therefore not part of this class’s public contract.

## Output is a function of state

The output is

$$
y(t) = \phi(x(t)),
$$

where `phi` is the selected activation/output function. The output need not equal the state. In the classical model, a bounded piecewise-linear output function is central; CELNN also exposes identity, tanh, sigmoid, sign, ReLU-like, and saturated alternatives for computational experiments.

Feedback uses `y(x)`. Consequently, changing the activation changes the vector field even if every template coefficient is unchanged.

## A field is many coupled cells

For a regular array, every position represents one cell state. A one-dimensional signal of length `N` represents `N` cells; a two-dimensional `H × W` array represents `H × W` cells. Direct coupling remains local because templates have finite extent.

This local representation does not make cells independent. After one cell changes, its output can change the derivative of its neighbors; those changes can then influence further neighbors later. The global trajectory is the result of repeated local propagation.

## Representation versus meaning

Array shape, dtype, and backend are numerical representation choices. State, input, output, and neighborhood are model concepts. Keeping that distinction explicit is important when moving between NumPy, CuPy, and PyTorch execution paths.

Next: {doc}`equation` derives the normalized equation used throughout this library.
