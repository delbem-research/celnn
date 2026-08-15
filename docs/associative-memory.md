# Delta-Hebbian key--value memory

`DeltaHebbianMemory` is a transient associative memory for PyTorch models. It
does not own conversation state or projections: callers explicitly create,
pass, replace, reset, detach, and serialize policy around the state.

For a key `k`, value `v`, and memory matrix `M`, the local rule computes:

```text
prediction = M k
error = v - prediction
M_next = retention * M + learning_rate * error k^T
```

The error term corrects an existing association instead of adding the same
outer product indefinitely. Keys and queries are L2-normalized by default.

```python
import torch
from celnn import DeltaHebbianMemory, DeltaHebbianRule

rule = DeltaHebbianRule(learning_rate=0.2, retention=0.99)
memory = DeltaHebbianMemory(
    key_size=32,
    value_size=32,
    rule=rule,
    detach_updates=False,
    memory_limit=1.0,
)
state = memory.new_state(batch_size=4, like=torch.zeros(1))

state = memory.write(state, key, value)
retrieved = memory.read(state, query)
state = state.reset()
```

`learning_rate` and `retention` can be tensors with shape `(batch,)`. This lets
a separate controller learn when to write and forget without coupling that
policy to the reusable memory rule.

The memory is constant-size: a key width `k` and value width `v` require `k*v`
dynamic scalars per sequence, independent of sequence length. With both widths
set to 32, that is 1,024 scalars, or 2 KiB in FP16.
