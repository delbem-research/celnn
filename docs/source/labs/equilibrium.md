---
file_format: mystnb
kernelspec:
  name: python3
---

(equilibrium-lab)=
# Equilibrium lab: small step, large residual

This lab tests a narrow scientific claim:

> A small state increment does not, by itself, prove that the continuous
> dynamical system is close to equilibrium.

For the one-cell system

$$
\dot{x} = -x + 1,
$$

an equilibrium satisfies $-x + 1 = 0$, so $x^* = 1$.

## Reference trajectory

The analytical solution with $x(0)=0$ is

$$
x(t) = 1 - e^{-t}.
$$

The first experiment compares ten explicit Euler steps from the public CELNN API
with that exact result.

```{code-cell} ipython3
import math

import numpy as np

from celnn import CellularNetwork, SimulationConfig


def make_network() -> CellularNetwork:
    return CellularNetwork(
        input=np.zeros(1),
        initial_state=np.zeros(1),
        feedback=np.zeros(3),
        control=np.zeros(3),
        bias=1.0,
        activation="identity",
    )


net = make_network()
dt = 0.1
for _ in range(10):
    state = net.step(dt)

exact = 1.0 - math.exp(-1.0)
error = abs(float(state[0]) - exact)
state, exact, error
```

```{code-cell} ipython3
assert error < 0.025
```

## Falsifying increment-only equilibrium detection

Now take a single public Euler step with a very small timestep.

```{code-cell} ipython3
net.reset()
dt = 1e-7
initial_state = net.state.copy()
next_state = net.step(dt)
state_increment = float(np.max(np.abs(next_state - initial_state)))
residual = float(np.max(np.abs(net.derivative(next_state))))
state_increment, residual
```

The increment is below $10^{-6}$ only because it is multiplied by the tiny
timestep. The vector field still has magnitude close to one, so the state is
not close to satisfying $f(x)=0$.

The falsifier targets mathematical properties of the example, not a temporary
implementation detail:

```{code-cell} ipython3
assert state_increment < 1e-6
assert residual > 0.9
```

Reversing the second assertion to `assert residual < 1e-3` must make notebook
execution fail. The documentation build is configured to elevate unexpected
execution errors to build failures.

## Current CELNN diagnostic

The current public {py:class}`celnn.SimulationResult` can be inspected
separately. This observation is versioned library behavior; it is deliberately
not the oracle for the mathematical claim above.

```{code-cell} ipython3
net.reset()
result = net.run(SimulationConfig(t_end=dt, dt=dt))
result.convergence
```

## How do we know?

The primary evidence is analytical: at the first Euler state
$x_1=10^{-7}$, the residual is $|-x_1+1|=0.9999999$. The executable assertions
compare the experiment with that prediction.

A future redesign of convergence/stability diagnostics should be proved in the
production test suite with an explicit scientific contract. This lab should not
freeze the current `approx_converged` heuristic as immutable behavior.
