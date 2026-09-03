# Associative memory and normalized fields

CELNN’s associative-memory APIs combine several ideas that should be kept historically and mathematically distinct: matrix key–value recall, error-correcting writes, fast transient state, and normalized positive-feature retrieval.

## Correlation-matrix lineage

Kohonen’s correlation matrix memory stores associations between key components and data components by accumulating their products; recall applies the resulting matrix-like association to a key. See {ref}`kohonen-1972`.

This supplies a clear lineage for representing key–value associations in a matrix. CELNN’s `DeltaHebbianMemory` uses the same broad matrix-memory geometry but a different write rule.

## Error-correcting write in CELNN

For normalized key `k`, target value `v`, and memory `M`, the implemented rule computes

$$
\hat v=Mk,
$$

$$
e=v-\hat v,
$$

$$
M_{next}=\rho M+\eta e k^T.
$$

The error term corrects the currently retrieved value rather than simply adding the target outer product indefinitely.

This documentation intentionally describes the equation from CELNN’s implementation. The available source set does not contain the primary Widrow–Hoff 1960 paper, so no indirect historical attribution is used.

## Normalized positive-feature retrieval

The associative field stores, for each cell, a numerator matrix `M` and a normalizer vector `s`. CELNN uses

$$
\phi(z)=\operatorname{elu}(z)+1
$$

and reads

$$
r(q)=\frac{M\phi(q)}{s^T\phi(q)+\epsilon}.
$$

Katharopoulos and colleagues use positive kernel feature maps, recurrent numerator/normalizer accumulators, and specifically `elu(x)+1` in their linear-attention formulation; see {ref}`katharopoulos-et-al-2020`.

That paper does **not** define CELNN’s `NormalizedDeltaHebbianField`. CELNN combines normalized feature retrieval with its own local error-correcting write construction.

## Why numerator and normalizer move together

Changing the normalizer changes the denominator of a future read. A write that updated only the numerator by a naive target outer product would not in general move the normalized response by the intended amount.

CELNN therefore computes a correction against the simultaneously updated normalizer so that the local normalized response to the written key is moved toward the target by the requested rate.

When a memory limit is configured, numerator and normalizer are rescaled together per cell. Because both scale by the same positive factor, their normalized ratio is preserved apart from the additive epsilon term.

## Storage topology is separate from spatial propagation

Each associative-field cell owns local memory, but the module does not prescribe how those cells communicate. The explicit state can be propagated by a CELNN, another grid operator, a graph process, or not propagated at all.

This separation prevents a specific memory mechanism from silently redefining the topology of the model using it.
