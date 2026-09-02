# Top-level public API

This page represents every symbol intentionally exported through `celnn.__all__`
exactly once. Optional Torch-backed symbols are structurally renderable in the
base documentation build; executable claims about them require a real Torch
environment.

```{py:currentmodule} celnn
```

## Core simulation

```{autoclass} celnn.CellularNetwork
:members:
```

```{autoclass} celnn.SimulationConfig
:members:
```

```{autoclass} celnn.SimulationResult
:members:
```

```{autoclass} celnn.Template
:members:
```

```{autoclass} celnn.TemplateRegistry
:members:
```

## Activation functions

```{autofunction} celnn.identity
```

```{autofunction} celnn.piecewise_linear
```

```{autofunction} celnn.relu_activation
```

```{autofunction} celnn.saturated_linear
```

```{autofunction} celnn.sign_activation
```

```{autofunction} celnn.sigmoid_activation
```

```{autofunction} celnn.tanh_activation
```

## Differentiable CELNN

Requires the `torch` optional capability for real execution.

```{autoclass} celnn.DifferentiableCellularNetwork
:members:
```

## Plasticity

These APIs require the `torch` optional capability for real execution.

```{autoclass} celnn.PlasticityState
:members:
```

```{autoclass} celnn.PlasticityRule
:members:
```

```{autoclass} celnn.HebbianRule
:members:
```

```{autoclass} celnn.OjaRule
:members:
```

```{autoclass} celnn.Plasticity
:members:
```

```{autoclass} celnn.PlasticLinear
:members:
```

## Associative memory

These APIs require the `torch` optional capability for real execution.

```{autoclass} celnn.AssociativeMemoryState
:members:
```

```{autoclass} celnn.DeltaHebbianRule
:members:
```

```{autoclass} celnn.DeltaHebbianMemory
:members:
```

```{autoclass} celnn.AssociativeFieldState
:members:
```

```{autoclass} celnn.NormalizedDeltaHebbianField
:members:
```
