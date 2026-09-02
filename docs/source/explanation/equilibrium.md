(equilibrium-explanation)=
# Equilibrium is a property of the vector field

For a continuous dynamical system

$$
\dot{x} = f(x),
$$

an equilibrium $x^*$ satisfies

$$
f(x^*) = 0.
$$

This definition belongs to the continuous system. It does not depend on which
time integrator is used to approximate a trajectory.

## Why a small step is not enough

Explicit Euler advances a state by

$$
x_{n+1} = x_n + \Delta t\,f(x_n).
$$

Therefore

$$
|x_{n+1} - x_n| = \Delta t\,|f(x_n)|.
$$

A sufficiently small $\Delta t$ can make the state increment arbitrarily small
without making the vector-field residual $|f(x_n)|$ small. A criterion based
only on the final state increment can therefore confuse fine discretization
with physical stationarity.

The [executable equilibrium lab](../labs/equilibrium.md) constructs this
counterexample using the public CELNN API and checks the stable mathematical
properties directly.

## Library behavior versus mathematical truth

{py:attr}`SimulationResult.convergence` is an established public interface in
CELNN 1.0. Its current fields are useful diagnostics, but their present
implementation is not the definition of equilibrium. See
{py:class}`SimulationResult` in the generated API reference and the
[scientific ownership map](../internals/scientific-ownership.md) for where the
relevant implementation and evidence live.
