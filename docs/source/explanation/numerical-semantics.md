# Numerical semantics and solver trade-offs

A solver answers: **how will this continuous vector field be approximated on a computer?** It does not answer what the underlying CELNN model is.

## Explicit Euler exposes the vector field directly

Euler uses

$$
x_{n+1}=x_n+dt\,f(x_n).
$$

Its strengths are inspectability, low overhead, and a direct relationship between the derivative and the increment. Its weakness is that accuracy and numerical stability can degrade rapidly when the step is too large relative to the dynamics.

## Semi-implicit Euler isolates the linear leak

CELNN writes

$$
f(x)=-x+d(x)
$$

and advances with

$$
x_{n+1}=\frac{x_n+dt\,d(x_n)}{1+dt}.
$$

Only the leak is implicit. The nonlinear/recurrent drive is still taken from the current state. Calling this method “implicit Euler” without the qualifier would overstate what is solved implicitly.

## `solve_ivp` is an independent integration path, not an oracle by definition

SciPy’s `solve_ivp` provides mature adaptive ODE machinery and is useful for differential comparisons against fixed-step methods. But agreement with one numerical library is still numerical evidence. It does not replace an analytic result where one is available.

The strength of a differential comparison comes from implementation independence: different numerical algorithms that agree under refinement make a shared coding error less likely.

## Step refinement is evidence about discretization

For a quantity of interest `Q`, compare results under a sequence such as

$$
dt,\;dt/2,\;dt/4.
$$

If the quantity stabilizes at the expected rate for the chosen method, that supports the claim that discretization error is controlled in that regime. Merely choosing a small decimal value for `dt` does not provide the same evidence.

## Tolerances need a scale

A tolerance such as `1e-6` is meaningful only relative to dtype, state scale, method, and the property being tested. A tolerance should follow from an error model, an independent comparison, or observed refinement behavior—not be widened until a test passes.

## Solver diagnostics and scientific claims have different owners

A solver may expose warnings or heuristic convergence fields for convenience. A scientific statement such as “the final state is an equilibrium to tolerance ε” should instead define the residual and tolerance explicitly and evaluate that quantity.

This ownership rule lets diagnostic implementations improve later without silently changing the meaning of published science.
