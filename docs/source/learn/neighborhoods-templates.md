# Neighborhoods and templates

A Cellular Neural Network is *cellular* because direct interaction is local. In the original two-dimensional definition, Chua and Yang define an `r`-neighborhood around a cell and illustrate radii 1, 2, and 3 as 3×3, 5×5, and 7×7 neighborhoods. See {ref}`chua-yang-1988-theory`.

CELNN generalizes the same local-stencil idea to regular arrays of matching dimensionality.

## Relative offsets

For a one-dimensional three-point template

```text
[left, center, right]
```

the center coefficient acts on the current cell and the other coefficients act on immediate neighbors.

For a two-dimensional 3×3 template

```text
w00  w01  w02
w10  w11  w12
w20  w21  w22
```

`w11` is the current-cell coefficient. The remaining entries correspond to relative spatial offsets around it.

## Feedback and control share geometry, not meaning

`A` and `B` use the same neighborhood geometry but act on different fields:

$$
A * y(x) \qquad\text{versus}\qquad B * u.
$$

A large center coefficient in `A` changes recurrent self-coupling. The same coefficient in `B` changes how strongly the current input cell drives the derivative. Similar matrices can therefore have different dynamical meaning depending on which operator owns them.

## The `Template` representation

{py:class}`celnn.Template` packages a feedback array, control array, bias, optional initial state, and descriptive metadata. Its validation enforces the representational invariants required by the regular-grid stencil model, including compatible feedback/control shapes and a well-defined center.

A `Template` is a reusable model component. It is not a guarantee that a particular task will work, converge, or be stable. Those are behavioral claims that require analysis or evidence.

## Template extent and model dimensionality

The reference network validates template geometry against the regular-grid topology. A one-dimensional field uses one-dimensional templates; a two-dimensional field uses two-dimensional templates. Finite odd extents give each axis an unambiguous center coefficient.

## Templates are not merely filters

For the control term `B * u`, it is often useful to recognize familiar smoothing or contrast stencils. But the full system is not generally equivalent to applying a fixed linear filter once. Feedback recirculates the activated state, and the result depends on time evolution.

That is why template interpretation must include activation, bias, initial state, boundary semantics, and numerical integration rather than only the coefficient matrix.

## Reuse and provenance

CELNN ships demonstrative templates for image processing, logic-like behavior, diffusion, and pattern experiments. Built-ins should be treated according to their metadata and names: examples marked as demonstrations are starting points, not claims of optimality or canonical scientific parameters.

Next: {doc}`boundaries-spatial-coupling` shows why a finite template is not a complete spatial operator until edge behavior is specified.
