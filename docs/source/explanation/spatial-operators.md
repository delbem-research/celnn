# Templates as spatial operators

A CELNN template is a finite collection of coefficients indexed by relative position. Applying it means taking a weighted sum of aligned neighboring values. This is most precisely described in the library as a **stencil operator**.

## Translation-invariant interior operator

For a one-dimensional three-point stencil with coefficients $(a_{-1},a_0,a_{+1})$, the interior aggregation at cell `i` is

$$
(A*v)_i = a_{-1}v_{i-1}+a_0v_i+a_{+1}v_{i+1}.
$$

The same coefficient pattern is reused at every interior location. In multiple dimensions, relative offsets generalize in the obvious Cartesian way.

CELNN’s shared backend implementation realizes this directly by padding the field and accumulating one weighted shifted window per template offset. The implementation does not require a dense interaction matrix.

## Correlation/convolution terminology

Classical CNN literature often writes the translation-invariant template action using convolution notation, and frequency-domain analysis treats it as a spatial filtering operator. CELNN’s implementation contract is stencil alignment by relative offsets; documentation should therefore reason from those offsets rather than rely on an implicit kernel-flipping convention.

For symmetric templates the distinction is invisible. For asymmetric templates it matters.

## Feedback makes a spatial operator part of a dynamical operator

The control term $B*u$ applies a local spatial operator to a fixed input. The feedback term $A*y(x)$ applies a local spatial operator to the current nonlinear output and then feeds the result back into the state derivative.

Therefore

```text
spatial stencil + recurrence + nonlinearity + leak
```

is a dynamical operator, not merely a static filter.

## Linearized analysis is useful but conditional

When the output function is locally linear and the operating region remains within that regime, the dynamics can be analyzed using linear spatial-filter tools. Crounse and Chua use a spatial-frequency representation to explain filtering and pattern selection in the simple CNN; see {ref}`crounse-chua-1995`.

That analysis is powerful because it identifies modes and gains. It is also conditional: once cells move through nonlinear regions, the linearized transfer function is no longer a complete description.

## Familiar stencils are clues, not proofs

A discrete Laplacian-like stencil suggests diffusion-related behavior because of its spatial operator structure. But the full CELNN behavior still depends on how that stencil enters `A` or `B`, the sign and scaling, activation, bias, boundary, and time evolution.

The correct reasoning path is from coefficients to operator, from operator to vector field, and only then from vector field to expected behavior.
