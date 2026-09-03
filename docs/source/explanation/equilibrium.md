# Equilibrium is a property of the vector field

For a continuous-time system

$$
\dot{x}=f(x),
$$

an equilibrium $x^*$ satisfies

$$
f(x^*)=0.
$$

For the CELNN model implemented by the library,

$$
f(x)=-x+A*y(x)+B*u+z.
$$

Therefore a candidate equilibrium should be judged by the **vector-field residual**, not merely by how little one numerical step changed the state.

## Why the last increment can be misleading

Explicit Euler gives

$$
\Delta x_n=x_{n+1}-x_n=dt\,f(x_n).
$$

A threshold on $\|\Delta x_n\|$ is therefore a threshold on `dt * residual`. If `dt` is made arbitrarily small, the increment can be made arbitrarily small without moving the state closer to a root of the vector field.

This is the mechanism demonstrated in {doc}`../labs/equilibrium`.

## Residual and increment answer different questions

A small residual says the continuous model is nearly stationary at the evaluated state.

A small increment says the chosen discrete algorithm moved little over its last step. That may happen because the residual is small, because the step is small, because the method is damping strongly, or for another solver-specific reason.

Both quantities can be useful. They should not be given the same name.

## Stability requires perturbation reasoning

Even an exact equilibrium may be unstable. Stability concerns how trajectories behave after nearby perturbations; it cannot be established by evaluating only `f(x*)=0`.

The classical Chua–Yang theory proves convergence/stability statements for the specific circuit family and assumptions analyzed there; see {ref}`chua-yang-1988-theory`. CELNN exposes a more general computational surface with multiple activations and numerical configurations, so those historical results are not promoted into universal software guarantees.

## A defensible numerical equilibrium report

For a numerical endpoint, report at least:

- the norm used for the vector-field residual;
- the residual value;
- dtype and solver;
- the integration/refinement procedure that produced the state;
- any state-increment diagnostic separately.

When an exact equilibrium is known, compare against it directly. When none is known, step refinement or an independent solver can add evidence, but the limitations should remain explicit.
