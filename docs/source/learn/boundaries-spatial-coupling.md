# Boundaries and spatial coupling

A stencil describes relative coefficients for neighbors that exist in the interior of a field. At an edge, some requested offsets fall outside the stored array. A boundary rule defines what values those offsets mean.

For a finite computational field, the effective spatial operator is therefore

$$
\boxed{\text{template coefficients} + \text{boundary semantics}.}
$$

## CELNN boundary modes

The reference network supports five public modes:

- `constant` — values outside the array are replaced by `boundary_value`;
- `wrap` — the domain is periodic;
- `reflect` — reflection excludes repeating the edge sample;
- `nearest` — the edge sample is extended outward;
- `mirror` — symmetric reflection includes the edge sample.

The implementation normalizes these public meanings across backend libraries even where NumPy and SciPy use different names for the two reflection conventions.

## Why boundaries change behavior

Consider the three-point operator

$$
[1, -2, 1].
$$

In the interior it measures a discrete second difference. At the first element, the left neighbor does not exist in storage. A zero constant boundary, a wrapped neighbor from the opposite end, and a reflected neighbor all produce different values. The coefficient array is unchanged; the operator is not.

This matters especially for recurrent feedback because an edge discrepancy changes a derivative, which changes future state, which can then propagate inward.

## Periodic versus finite domains

`wrap` is appropriate only when the modeled domain is intentionally periodic. It makes opposite edges adjacent in the numerical topology.

Reflection modes are often useful when a finite signal or image should not couple to an artificial constant exterior. They still impose a modeling assumption: reflected data are not neutral; they create a particular extension of the field.

`constant` is explicit and useful when the exterior has a meaningful fixed value. The chosen `boundary_value` is part of the model configuration.

## Boundary semantics are independent of solver choice

Euler, semi-implicit Euler, and SciPy `solve_ivp` all evaluate derivatives whose local aggregations use the configured boundary semantics. Changing the solver should change only how the same continuous vector field is approximated, not what neighboring values mean.

Next: {doc}`time-evolution` separates continuous dynamics from their discrete numerical approximation.
