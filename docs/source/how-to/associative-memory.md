# Use associative memory

The optional PyTorch API provides two related transient key–value memories: a batch of matrix memories and a field of local normalized memories.

## Delta-Hebbian matrix memory

```python
import torch
from celnn import DeltaHebbianMemory, DeltaHebbianRule

rule = DeltaHebbianRule(learning_rate=0.2, retention=0.99)
memory = DeltaHebbianMemory(key_size=32, value_size=24, rule=rule)
state = memory.new_state(batch_size=4, like=torch.zeros(1))

state = memory.write(state, key, value)
retrieved = memory.read(state, query)
```

The implemented rule predicts with the current matrix, computes `value - prediction`, and writes an error-correcting outer-product update. Keys and queries are L2-normalized by default.

The current source set does not contain the primary Widrow–Hoff 1960 paper, so this documentation does not make a second-hand historical attribution for this update.

## Normalized associative fields

```python
from celnn import NormalizedDeltaHebbianField

field = NormalizedDeltaHebbianField(key_size=16, value_size=16)
state = field.new_state(batch_size=4, cells=32, like=activity)
state = field.write(state, keys, values, mask=active_cells)
retrieved = field.read(state, queries)
```

Each cell owns a numerator matrix and a normalizer vector. The read is a normalized positive-feature response. `read_all` instead evaluates arbitrary queries against every memory cell.

## Keep accumulator precision explicit

`new_state(..., dtype=...)` can keep memory accumulators in a wider dtype than surrounding projected activity. Reads and writes execute their accumulations in the state dtype.

## Do not confuse storage topology with propagation topology

The associative field does not prescribe how memories move between cells. A caller may propagate its explicit state with a CELNN, grid operation, graph operation, or no spatial coupling at all. This separation is intentional.

For the scientific lineage from correlation-matrix recall through fast-weight and normalized-feature formulations, see the later associative-memory explanation. Relevant primary sources include {ref}`kohonen-1972` and {ref}`katharopoulos-et-al-2020`.
