---
file_format: mystnb
kernelspec:
  name: python3
---

(equilibrium-lab)=
# Lab: falsify false equilibrium

## Claim

A small state increment over one numerical step is not sufficient evidence that
a continuous CELNN is near equilibrium.

For the one-cell system

$$
\dot{x} = -x + 1, \qquad x(0)=0,
$$

the equilibrium is $x^*=1$. Near $x=0$, the physical residual remains close to
one.

## Prediction

With explicit Euler and $\Delta t=10^{-7}$, the first increment is below
$10^{-6}$ while the residual remains larger than $0.9$.

## Experiment

```{code-cell} ipython3
import numpy as np

from celnn import CellularNetwork

net = CellularNetwork(
    input=np.zeros(1),
    initial_state=np.zeros(1),
    feedback=np.zeros(3),
    control=np.zeros(3),
    bias=1.0,
    activation="identity",
)

initial_state = net.state.copy()
dt = 1e-7
next_state = net.step(dt)

state_increment = float(np.max(np.abs(next_state - initial_state)))
residual = float(np.max(np.abs(net.derivative(next_state))))

state_increment, residual
```

The falsifier targets mathematical properties of the example, not a temporary
implementation detail:

```{code-cell} ipython3
assert state_increment < 1e-6
assert residual < 1e-3
```

Reversing the second assertion to `assert residual < 1e-3` must make notebook
execution fail. The documentation build is configured to elevate unexpected
execution errors to build failures.

## Current CELNN diagnostic

The current public result can be inspected separately. This observation is
versioned library behavior; it is deliberately not the oracle for the
mathematical claim above.

```{code-cell} ipython3
from celnn import SimulationConfig

net.reset()
result = net.run(SimulationConfig(t_end=dt, dt=dt))
result.convergence
```

## How do we know?

The primary evidence is analytical: at the first Euler state
$x_1=10^{-7}$, the residual is $|-x_1+1|=0.9999999$. The executable assertions
compare the library experiment with that prediction.

A future redesign of convergence/stability diagnostics should be proved in the
production test suite with an explicit scientific contract. This lab should not
freeze the current `approx_converged` heuristic as immutable behavior.
