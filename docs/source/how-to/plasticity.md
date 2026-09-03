# Add fast-weight plasticity

CELNN’s optional plasticity primitives are reusable PyTorch components. They are deliberately independent of the regular-grid CELNN dynamics.

Install the capability:

```bash
python -m pip install "celnn[torch]"
```

## Build a plastic linear layer

```python
from celnn import HebbianRule, Plasticity, PlasticLinear

plasticity = Plasticity(
    HebbianRule(learning_rate=0.02, decay=0.99),
    alpha=0.5,
    detach_updates=True,
    memory_limit=1.0,
)
layer = PlasticLinear(128, 128, plasticity)
state = layer.new_state(batch_size=4)

output, next_state = layer(input_batch, state, update=True)
```

The caller owns `state`. This makes sequence boundaries, reset policy, concurrent sessions, and graph truncation explicit.

## Choose a rule

`HebbianRule` applies a decayed mean pre/post outer-product update. `OjaRule` adds a local output-energy term that counteracts unbounded Hebbian growth. The scientific lineage of Oja’s stabilizing rule is documented in {ref}`oja-1982`.

## Control gradient history

By default, `Plasticity(detach_updates=True)` detaches newly computed fast memory from autograd. This is useful for online state where retaining the complete historical graph would be undesirable.

Set `detach_updates=False` only when the objective requires differentiating through the plastic update history.

## Compose slow and fast weights

The effective per-sample weight is

$$
W_{\text{effective}} = W_{\text{slow}} + \alpha H_{\text{fast}}.
$$

`alpha` may be fixed or learnable. `memory_limit` is an independent hard bound on fast memory; it is not equivalent to Oja stabilization or decay.

Fast-changing weights as short-term memory have a longer history; {ref}`schmidhuber-1992` is one explicit fast-weight memory formulation. CELNN’s API and update ownership are its own design, not a reproduction of that architecture.
