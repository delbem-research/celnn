# What is a Cellular Neural Network?

A **Cellular Neural Network (CelNN)** is a continuous-time dynamical system built from many cells arranged over a spatial domain, with direct interaction restricted to local neighborhoods. Chua and Yang introduced the architecture as a large aggregate of regularly spaced cells whose direct communication is local; cells that are not directly connected can still influence one another indirectly as dynamics propagate through neighboring cells. See {ref}`chua-yang-1988-theory`.

This documentation uses **CelNN** rather than the historically common abbreviation **CNN** to avoid confusion with convolutional neural networks. The two model families are not the same thing.

## The essential structure

A cell has three conceptually distinct quantities:

- a **state** `x`, which evolves in time;
- an **input** `u`, supplied from outside the state dynamics;
- an **output** `y(x)`, obtained by applying an output/activation function to the state.

Cells exchange information through local coefficients. In the normalized computational model implemented by `celnn`, the field dynamics are

$$
\frac{dx}{dt} = -x + A * y(x) + B * u + z,
$$

where `A` is feedback from neighboring outputs, `B` controls how the external input enters, `z` is a bias, and `*` denotes local neighborhood aggregation.

The important idea is not any particular image-processing template. It is the combination of **continuous state**, **local coupling**, **nonlinearity**, and **time evolution**.

## Local interaction, global consequences

A cell only reads a bounded neighborhood directly. Nevertheless, repeated evolution lets information propagate across the domain. This is why a local rule can produce a large-scale response: smoothing, sharpening, thresholded regions, traveling influence, or more complicated spatial organization can emerge from repeated local interactions.

The original literature focused heavily on analog circuits and image processing. The `celnn` package keeps the mathematical local-dynamics idea while representing fields as numerical arrays and integrating the resulting ODE computationally. It supports regular one- and multi-dimensional arrays rather than exposing the original circuit elements as the user abstraction.

## What a CelNN is not

A CelNN is not a one-shot convolution. A template aggregation contributes to the **derivative** of a state that then evolves through time. The same spatial stencil can therefore behave differently when feedback strength, activation, bias, initial state, boundary handling, solver, or time step changes.

A CelNN is also not a cellular automaton. Both use local neighborhoods, but the classical Cellular Neural Network is a continuous-time, continuous-state dynamical system. Chua and Yang explicitly contrast this continuous-time behavior with the discrete-time dynamics of cellular automata in {ref}`chua-yang-1988-theory`.

## Where to go next

Read {doc}`mental-model` for the smallest useful conceptual diagram, then {doc}`first-system` to construct and run one deterministic system with the public CELNN API.
