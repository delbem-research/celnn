# Modular Hebbian plasticity

The optional PyTorch API separates four concerns so plasticity can be reused in
CelNNs, recurrent models, memory layers, or ordinary feed-forward networks:

- `PlasticityState` owns transient, per-sample fast weights;
- `HebbianRule` and `OjaRule` implement local update equations;
- `Plasticity` controls composition, graph retention, and stability limits;
- `PlasticLinear` is a ready-to-use layer built from those primitives.

Importing `celnn` does not import PyTorch. These names are resolved lazily and
require the `torch` extra only when used.

## Functional state

Plastic memory is passed in and returned explicitly. Modules do not hide a
conversation or sequence state, which makes reset, batching, serialization,
truncated backpropagation, and concurrent sessions caller-controlled.

```python
from celnn import HebbianRule, PlasticLinear, Plasticity

plasticity = Plasticity(
    HebbianRule(learning_rate=0.02, decay=0.99),
    alpha=0.5,
    detach_updates=True,
    memory_limit=1.0,
)
layer = PlasticLinear(128, 128, plasticity)
state = layer.new_state(batch_size=4)

output, state = layer(input_batch, state, update=True)
state = state.detach()
state = state.reset()
```

Each batch element owns a matrix with shape `(output, input)`. Sample axes
between batch and features are averaged when calculating a local update, so a
sequence shaped `(batch, time, features)` produces one update per sequence and
never mixes memories across the batch.

## Rules

The decayed Hebbian rule is:

```text
H_next = decay * H + learning_rate * mean(post outer pre)
```

Oja's rule adds a local stabilizing term:

```text
H_next = decay * H
       + learning_rate * (mean(post outer pre) - mean(post²) * H)
```

Custom rules only need to implement the `PlasticityRule` callable protocol:

```python
class MyRule:
    def __call__(self, pre, post, memory):
        return next_memory
```

## Slow and fast weights

`Plasticity.effective_weight` composes parameters without modifying the model's
checkpointed weights:

```text
effective = slow_weight + alpha * fast_memory
```

By default, updates are detached from autograd. This is appropriate for online
inference and prevents an indefinitely growing graph. Set
`detach_updates=False` when meta-learning through the plastic update itself.
`alpha` can also be learned with `learnable_alpha=True`.

`memory_limit` provides a hard symmetric bound. Oja normalization, decay, and
hard limits are independent mechanisms and can be combined.

## CelNN integration

Plastic channel mixing belongs outside the canonical CelNN ODE. A caller can
apply a `PlasticLinear` across the channel axis and supply its output through
`DifferentiableCellularNetwork.step(..., extra_drive=...)`. This preserves the
single canonical derivative and keeps cross-channel computation explicit.
