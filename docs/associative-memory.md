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

## Normalized associative fields

`NormalizedDeltaHebbianField` generalizes the key--value memory to a field of
independent local memories. Each cell owns a numerator matrix `M_i` and a
positive normalizer vector `s_i`:

```python
from celnn import NormalizedDeltaHebbianField

field = NormalizedDeltaHebbianField(key_size=16, value_size=16)
state = field.new_state(batch_size=4, cells=32, like=activity)
state = field.write(state, keys, values, mask=active_cells)
retrieved = field.read(state, queries)
```

Callers may keep recurrent accumulators in a wider dtype than their learned
projections:

```python
state = field.new_state(
    batch_size=4,
    cells=32,
    like=activity,
    dtype=torch.float32,
)
```

Reads and writes explicitly disable mixed-precision autocast and convert
keys, values, gates, and queries to the state dtype. Thus `M`, `s`, numerator,
denominator, and Delta-Hebbian corrections remain FP32 while the surrounding
network can use BF16.

The read is `M_i phi(q_i) / (s_i^T phi(q_i) + epsilon)`, with the strictly
positive feature map `phi(z) = elu(z) + 1`. A write moves that normalized
response by `learning_rate * (value - prediction)`. The corresponding local
outer-product correction accounts for the simultaneous change in `s_i`, so a
larger denominator cannot make an already-correct association worse.

When `memory_limit` is set, the implementation rescales `M_i` and `s_i`
together per cell instead of clamping only the numerator. Their normalized
read is therefore preserved while both pieces of transient state stay bounded.

The library keeps spatial propagation separate from associative storage. A
caller can therefore diffuse both `state.memory` and `state.normalizer` over
a one-dimensional lattice, image grid, or arbitrary graph without coupling
the primitive to a particular topology.
