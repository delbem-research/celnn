# Use and create templates

{py:class}`celnn.Template` is the reusable representation of feedback `A`, control `B`, bias `z`, optional initial state, and descriptive metadata.

## Create a template

```python
from celnn import Template

template = Template(
    name="three_point_example",
    feedback=[0.1, 1.0, 0.1],
    control=[0.0, 1.0, 0.0],
    bias=0.0,
    description="Small one-dimensional demonstration",
    tags=["1d", "demo"],
).validate()
```

Validation checks representational consistency. It does not prove task quality or stability.

## Prefer a registry when names are part of your workflow

```python
from celnn import TemplateRegistry

registry = TemplateRegistry()
registry.register(template)
copy_for_use = registry.get("three_point_example")
```

The registry stores copies and returns copies, preventing accidental mutation of the registered object. Registering the same name again requires `overwrite=True`.

## Inspect built-ins

```python
from celnn.templates import builtin_templates

registry = builtin_templates()
print(registry.names())
```

Built-ins include image-processing, logic-like, diffusion, and pattern demonstrations. Names containing `DEMO` should be read literally: they are demonstrative configurations, not claims of optimality.

## Design a new template systematically

Do not tune coefficients in isolation. A useful workflow is:

1. state the intended behavior in terms of the CELNN equation;
2. decide what should come from recurrent feedback `A` and what should come from input control `B`;
3. choose the smallest neighborhood consistent with the mechanism;
4. choose activation, bias, boundary semantics, and initial state;
5. choose an integration method and conservative numerical resolution;
6. inspect trajectories and vector-field residuals, not only the final image or signal;
7. turn any required production property into a test or independent oracle.

For the meaning of the coefficients, see {doc}`../learn/neighborhoods-templates`. For automated search, see {doc}`train-ga` or {doc}`differentiable`.

## Probe the mechanism before tuning it

The old documentation offered fixed coefficient and timestep recipes for broad task classes. Those numbers are not universal CELNN guarantees, so the safer workflow is to diagnose the operator with simple inputs before tuning a real dataset.

Useful probes include:

- a constant field, which exposes unintended bias or boundary effects;
- a single impulse, which makes the local spatial response easy to inspect;
- a step or edge, which reveals smoothing, sharpening, or directional response;
- a checkerboard, which stresses high-spatial-frequency behavior;
- a deterministic sine wave, which is useful for one-dimensional filtering experiments.

Change one modeling choice at a time. If a coefficient change appears helpful only for one timestep or one boundary mode, treat that as evidence to investigate rather than as a transferable template rule.

## Troubleshoot from the failing property

If behavior is wrong only near the edge of the domain, inspect the boundary operator before changing interior coefficients. If the trajectory changes materially under timestep refinement, investigate discretization error before retuning the template. If output looks clipped, inspect both the internal state and the activation instead of assuming the stencil is wrong. If the last state increment is small, inspect the vector-field residual before calling the system converged.

When a design becomes difficult to explain, reduce it to the smallest neighborhood and simplest activation that still reproduces the relevant behavior. Add complexity only after the simpler mechanism has been falsified.
