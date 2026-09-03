# The Cellular Neural Network equation

CELNN uses the following normalized continuous-time model as its canonical computational definition:

$$
\boxed{\dot{x} = -x + A * y(x) + B * u + z}
$$

This is a nondimensionalized software model of the same structural terms used in the classical Cellular Neural Network equations: decay, output feedback, input control, and bias/current drive. The original circuit paper includes physical component parameters; the library intentionally exposes the normalized dynamical form instead. See {ref}`chua-yang-1988-theory`.

## Decompose the vector field

Define the non-decay drive

$$
d(x) = A * y(x) + B * u + z.
$$

Then

$$
\dot{x} = -x + d(x).
$$

This decomposition is exactly the one used by the implementation. It is also useful numerically because the semi-implicit Euler step treats `-x` implicitly while evaluating the drive explicitly.

## The decay term

The term `-x` pulls the state toward zero in the absence of all other drive:

$$
\dot{x}=-x.
$$

This local leak is stabilizing in isolation, but it does **not** guarantee stability of the complete coupled nonlinear system. Strong recurrent feedback can dominate it, and numerical discretization adds a separate stability question.

## Feedback: `A * y(x)`

`A` aggregates the output field over a finite neighborhood. Because the output depends on state, this is the recurrent term. Chua and Yang refer to the corresponding classical coefficients as a feedback operator. See {ref}`chua-yang-1988-theory`.

The symbol `*` in this documentation means CELNN’s local stencil aggregation. It is spatial, not multiplication of two scalars.

## Control: `B * u`

`B` aggregates the fixed external input. In contrast with feedback, this term does not depend on the current state when `u` is fixed. It can represent direct local filtering or forcing of the state dynamics.

## Bias: `z`

`z` adds a broadcastable local drive. It changes the equilibrium equation and can move the operating point of a nonlinear activation.

## Equilibrium equation

A state `x*` is an equilibrium of the continuous model when

$$
0 = -x^* + A * y(x^*) + B * u + z.
$$

Equivalently,

$$
x^* = A * y(x^*) + B * u + z.
$$

This definition involves the vector field, not a numerical step size. That distinction becomes central in {doc}`equilibrium-convergence-stability`.

## Implementation ownership

{py:meth}`celnn.CellularNetwork.derivative` is the public way to evaluate the canonical derivative at a state. Internally, `celnn.core.dynamics` owns the decomposition into feedback, control, drive, and derivative. Time-stepping formulas are separate owners; they approximate this equation but do not redefine it.

Next: {doc}`neighborhoods-templates` explains what the local operators `A` and `B` mean spatially.
