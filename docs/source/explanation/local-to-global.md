# From local rules to global behavior

The defining structural constraint of a Cellular Neural Network is local direct interaction. Chua and Yang describe cells as directly connected only to neighboring cells while noting that non-neighboring cells can still influence one another indirectly through propagation of the continuous-time dynamics. See {ref}`chua-yang-1988-theory`.

That statement contains the core mechanism behind global behavior.

## Influence propagates by repeated local dependence

Suppose cell `i` directly depends only on neighbors within radius one. At a given instant, its derivative can react only to that local neighborhood. After those neighbors change, however, their new outputs affect their own neighbors. Repetition extends the causal influence across the field without adding any direct long-range edge.

In a time-stepped picture:

```text
step 0: cell i affects immediate neighbors
step 1: those changes affect their neighbors
step 2: influence has reached a wider region
...
```

The exact propagation is not generally a finite-speed graph process because the underlying model is a coupled ODE, but the dependency structure remains local.

## Locality creates structure and scalability

For regular translation-invariant templates, the same small set of coefficients is reused throughout the interior. The number of cells may be large while the parameter description remains small. This locality is part of why Cellular Neural Networks were historically attractive for parallel analog implementations.

In software, the same structure leads naturally to stencil computation rather than a dense all-to-all interaction matrix.

## Global behavior is not encoded in one coefficient

A large-scale pattern cannot usually be attributed to one template entry in isolation. It emerges from the combined vector field:

- local feedback and control coefficients;
- activation regime;
- bias;
- input and initial state;
- boundary semantics;
- elapsed time.

For numerical experiments, the integration method and resolution additionally affect the observed approximation.

## Emergence does not remove the need for mechanism

Calling a pattern “emergent” should not be used as an explanation by itself. A useful explanation identifies the local mechanism that amplifies, damps, transports, or selects spatial modes and then connects that mechanism to the observed global structure.

The filtering and pattern-formation tutorial by Crounse and Chua performs exactly this kind of analysis for classes of CNN behavior using linearized spatial-frequency dynamics; see {ref}`crounse-chua-1995`.
