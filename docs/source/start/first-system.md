# Your first CELNN system

This example uses a one-dimensional input and a stable linear drive so that every component can be inspected without image I/O, randomness, optional dependencies, or a complicated template.

The feedback template is zero. The control template copies only the input at the current cell. With the identity activation, the model is therefore

$$
\dot{x} = -x + u.
$$

Starting from zero, each cell relaxes toward its corresponding input value.

```{code-cell} python
import numpy as np

from celnn import CellularNetwork, SimulationConfig

u = np.array([-1.0, -0.25, 0.5, 1.0])

network = CellularNetwork(
    input=u,
    feedback=np.array([0.0, 0.0, 0.0]),
    control=np.array([0.0, 1.0, 0.0]),
    bias=0.0,
    activation="identity",
    boundary="constant",
)

result = network.run(
    SimulationConfig(
        t_end=1.0,
        dt=0.05,
        solver="euler",
        return_trajectory=True,
    )
)

assert result.state.shape == u.shape
assert result.output.shape == u.shape
assert result.trajectory_state is not None
assert result.trajectory_state.shape[1:] == u.shape
```

## Read the construction from top to bottom

`input=u` fixes the external field. `feedback` and `control` are three-point local templates, so the middle coefficient refers to the current cell. Because all feedback coefficients are zero, neighboring outputs do not affect this example. Because the center control coefficient is one, each cell receives its own input value.

`boundary="constant"` still has a defined meaning, but it does not change this particular calculation because the only nonzero stencil coefficient is the center coefficient.

{py:class}`celnn.SimulationConfig` then asks the explicit Euler solver to approximate one second of continuous evolution with a step of `0.05`. The trajectory is stored because `return_trajectory=True`.

## Inspect state and output separately

With identity activation, state and output have the same numerical values, but they remain different concepts and separate fields in {py:class}`celnn.SimulationResult`. Other activations make that distinction visible immediately.

A useful next experiment is to replace `activation="identity"` with `"piecewise_linear"`, then increase the magnitude of `u`. The state may continue beyond the activation’s linear region while the output saturates.

## Do not infer convergence from one small step

A small difference between the last two discrete states is not, by itself, proof that the continuous vector field is near zero. The later unit {doc}`../learn/equilibrium-convergence-stability` and the executable {doc}`../labs/equilibrium` lab make that distinction explicit.
