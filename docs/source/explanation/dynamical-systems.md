# CELNNs as dynamical systems

A CELNN is best understood first as a dynamical system and only second as an image or signal operator. Once the input, templates, activation, bias, and boundary semantics are fixed, the network defines a vector field over the space of all cell states.

For a field with `N` scalar cells, the state can be viewed as a point in $\mathbb{R}^N$ even when the software stores it as a one- or multi-dimensional array. The canonical equation

$$
\dot{x}=f(x)=-x+A*y(x)+B*u+z
$$

defines a direction and rate of motion at every admissible state.

## A trajectory is an orbit of the vector field

Starting from an initial state $x(0)$, the continuous model determines a trajectory $x(t)$. The input `u` is part of the vector field in the reference `CellularNetwork`; it is not another state variable that evolves alongside `x`.

This distinction explains why initial state and input have different roles. Two experiments with the same input but different initial states can follow different transients. Two experiments with the same initial state but different inputs are different forced vector fields.

## Nonlinearity enters through the output map

The feedback term depends on $y(x)$ rather than directly on $x$. With a nonlinear bounded output map, the Jacobian and effective coupling change with state. Consequently, a coefficient matrix that appears smoothing-like near one operating point may participate in qualitatively different dynamics elsewhere.

The foundational circuit formulation is explicitly continuous-time and locally interconnected; see {ref}`chua-yang-1988-theory`. CELNN preserves that structural model while using numerical arrays rather than circuit voltages as its representation.

## Equilibria are fixed points of the continuous flow

An equilibrium satisfies $f(x^*)=0$. This is a property of the model. Stability asks what nearby trajectories do after perturbation. Basins of attraction ask which initial states approach which attractors.

A numerical endpoint is evidence about these properties only to the extent that the numerical method and diagnostic justify the inference. This is why CELNN separates the derivative from the solver implementation and why {doc}`equilibrium` treats residuals independently of state increments.

## Discrete simulation creates another dynamical map

A fixed-step numerical method defines a discrete map. Explicit Euler, for example, creates

$$
F_{dt}(x)=x+dt\,f(x).
$$

The discrete map approximates the continuous flow but is not identical to it. It can introduce numerical instability, alter transient detail, and make a small state increment simply because `dt` is small.

Scientific interpretation therefore follows the chain

```text
continuous definition → numerical representation → observed evidence.
```

Skipping a link in that chain is how implementation details get mistaken for mathematical results.
