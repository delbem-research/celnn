# Equilibrium, convergence, and stability

These terms answer different questions. Treating them as synonyms is a common source of incorrect scientific conclusions.

## Equilibrium belongs to the continuous vector field

For

$$
\dot{x}=f(x),
$$

a state `x*` is an equilibrium when

$$
f(x^*)=0.
$$

For CELNN,

$$
-x^* + A*y(x^*) + B*u + z = 0.
$$

This definition contains no numerical step size and no solver-specific threshold.

## A small state increment is solver-dependent

Explicit Euler gives

$$
x_{n+1}-x_n = dt\,f(x_n).
$$

Therefore

$$
\|x_{n+1}-x_n\| = dt\,\|f(x_n)\|
$$

for any homogeneous norm. A very small `dt` can make the discrete increment small even while the vector-field residual remains large.

That is a direct derivation from the Euler formula, not a corner case of CELNN.

For example, with

$$
\dot{x}=1-x,\qquad x(0)=0,
$$

and `dt = 10^{-7}`, the first Euler increment is `10^-7`, while the residual after that step is approximately `0.9999999`. The state barely moved because the time interval was tiny, not because it was near equilibrium.

## CELNN’s current convergence field is a diagnostic

The current solver stores `max_abs_state_delta`, the maximum absolute difference between the final two solver states, and sets `approx_converged` when that value is below `1e-6`.

This interface is useful for observing the current implementation. It is **not** a formal equilibrium certificate because it does not independently evaluate the final vector-field residual and its meaning depends on the solver’s last step.

The documentation deliberately does not assert that `approx_converged=True` proves mathematical convergence.

## Convergence needs a stated target

“Converged” can mean several things:

- a numerical sequence approaches a limit;
- a solver meets its internal error criterion;
- a state approaches an equilibrium of the vector field;
- an observable output stops changing at a chosen resolution.

A scientific claim should say which of these is intended and which quantity was measured.

## Stability is another property

Stability asks what happens to trajectories after perturbations. An equilibrium may exist but be unstable. A trajectory may approach one equilibrium from one region of state space and a different one from another region.

The foundational Chua–Yang paper proves stability properties for the specific circuit class and assumptions studied there; those theorems must not be generalized automatically to every activation, parameterization, or extension exposed by a software library. See {ref}`chua-yang-1988-theory`.

## Evidence hierarchy for an equilibrium claim

When possible, prefer evidence in this order:

1. an exact analytic solution or residual;
2. an independent numerical reference;
3. a differential comparison between implementations;
4. a property that follows from the governing equations;
5. a tolerance-based observation whose scale and rationale are explicit.

For a numerical endpoint, reporting both state increment and residual is substantially more informative than reporting only one.

The executable {doc}`../labs/equilibrium` lab demonstrates the false-convergence counterexample and protects the published claim without freezing the current diagnostic as a permanent scientific contract.
