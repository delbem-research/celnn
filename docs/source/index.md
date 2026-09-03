# CELNN Technical Knowledge System

`celnn` is a scientific Python library for Cellular Neural Networks: continuous-time, locally coupled nonlinear dynamical systems over regular arrays. This site separates learning, practical use, scientific explanation, exact API contracts, executable evidence, and implementation ownership so that each kind of knowledge has a clear home.

If you are new to Cellular Neural Networks, begin with **Start Here** and continue through **Learn** in order. If you already know the model and need to accomplish a task, use **Use CELNN**. Exact signatures and supported public objects belong in the generated API reference.

```{toctree}
:maxdepth: 2
:caption: Start Here

start/what-is-celnn
start/mental-model
start/first-system
```

```{toctree}
:maxdepth: 2
:caption: Learn

learn/cells-state-input-output
learn/equation
learn/neighborhoods-templates
learn/boundaries-spatial-coupling
learn/time-evolution
learn/equilibrium-convergence-stability
```

```{toctree}
:maxdepth: 2
:caption: Use CELNN

how-to/index
how-to/create-network
how-to/templates
how-to/activation-boundaries
how-to/run-simulation
how-to/inspect-results
how-to/data-artifacts
how-to/solvers-backends
how-to/train-ga
how-to/differentiable
how-to/plasticity
how-to/associative-memory
how-to/migrate-pycnn
```

```{toctree}
:maxdepth: 2
:caption: Explanation

explanation/dynamical-systems
explanation/local-to-global
explanation/spatial-operators
explanation/boundary-operators
explanation/numerical-semantics
explanation/equilibrium
explanation/diffusion-filtering-patterns
explanation/template-learning
explanation/plasticity-fast-weights
explanation/associative-memory
```

```{toctree}
:maxdepth: 2
:caption: Executable evidence

labs/equilibrium
```

```{toctree}
:maxdepth: 2
:caption: API Reference

reference/index
```

```{toctree}
:maxdepth: 2
:caption: Internals & Contributing

internals/architecture
internals/scientific-ownership
internals/dynamics
internals/stencils-topology-boundaries
internals/steppers-solvers
internals/backends-parity
internals/verification
internals/contribution-workflow
```

```{toctree}
:maxdepth: 1
:caption: Sources

bibliography
```

```{toctree}
:maxdepth: 1
:caption: Migration backlog

legacy
```
