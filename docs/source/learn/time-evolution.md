# Time evolution and numerical simulation

The CELNN equation is continuous in time. A computer produces a finite approximation to its trajectory. The mathematical model and the integration algorithm must therefore be kept distinct.

For

$$
\dot{x}=f(x),
$$

an exact trajectory would satisfy the differential equation for every time in an interval. CELNN provides discrete numerical solvers that sample or approximate that trajectory.

## Explicit Euler

For step size `dt`, explicit Euler uses

$$
x_{n+1}=x_n + dt\,f(x_n).
$$

This is CELNN’s default because its meaning is transparent and it uses only the derivative at the current state. It is also sensitive to step size: a method can be mathematically correct yet numerically inaccurate or unstable when `dt` is too large for a particular system.

## Semi-implicit Euler

Write the CELNN equation as

$$
\dot{x}=-x+d(x),
$$

where `d(x)=A*y(x)+B*u+z`. CELNN’s semi-implicit step treats only the linear leak implicitly:

$$
x_{n+1}=\frac{x_n+dt\,d(x_n)}{1+dt}.
$$

This is not a generic implicit solution of the nonlinear system. The recurrent drive is still evaluated from the current state.

## SciPy `solve_ivp`

When the optional SciPy dependency is installed, `solver="solve_ivp"` delegates integration of the flattened state to SciPy and reshapes states for derivative evaluation. It is useful as an independent integration path and for experiments that need SciPy’s ODE machinery.

CELNN does not claim that one solver is uniformly superior for every model. Choice depends on the experiment, required accuracy, cost, stiffness, and the evidence needed.

## `dt` has two roles in the public configuration

For Euler-family solvers, `dt` is the numerical step size. For the SciPy path, CELNN uses the configuration’s time points as requested evaluation points; SciPy controls its internal adaptive steps.

Consequently, statements about “the step size” must identify the solver being discussed.

## Accuracy and stability are different questions

A trajectory may remain bounded but be inaccurate. It may converge numerically to the wrong approximation. It may also diverge because a discrete method is unstable even when the underlying continuous system is stable.

A smaller `dt` often reduces explicit-Euler discretization error, but no universal CELNN threshold such as “`dt < c` is stable” follows from the API alone. Stability depends on the vector field being integrated.

## Stored trajectories are sampled evidence

With `return_trajectory=True`, {py:class}`celnn.SimulationResult` stores selected states and outputs. `store_every` changes what is retained, not which Euler steps are executed. A plotted trajectory is therefore a sampled record of the numerical computation, not the continuous solution itself.

Next: {doc}`equilibrium-convergence-stability` separates equilibrium of the vector field from numerical stopping diagnostics.
