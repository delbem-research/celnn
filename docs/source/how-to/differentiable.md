# Train a differentiable CELNN with PyTorch

Install PyTorch support:

```bash
python -m pip install "celnn[torch]"
```

{py:class}`celnn.DifferentiableCellularNetwork` is a PyTorch module that reuses CELNN’s canonical dynamics and time steppers while making one-dimensional template coefficients and bias PyTorch parameters or buffers.

## Construct and differentiate

```python
import torch
from celnn import DifferentiableCellularNetwork

network = DifferentiableCellularNetwork(
    radius=2,
    channels=16,
    dt=0.1,
    steps=10,
)

u = torch.randn(8, 64, 16)
y = network(u)
loss = y.square().mean()
loss.backward()
```

Use ordinary PyTorch optimizers on `network.parameters()`.

## Understand channel semantics

With no `channels` argument, the class follows the scalar one-dimensional contract. With explicit channels, templates are diagonal over channels: each channel has its own local spatial coefficients but the canonical CELNN dynamics do not mix channels internally.

If cross-channel mixing is required, make it explicit outside the canonical derivative or supply caller-owned `extra_drive` to the public Euler `step` path.

## Use causal neighborhoods

`causal=True` changes the one-dimensional template extent from symmetric offsets `[-radius, ..., radius]` to `[-radius, ..., 0]`. `shared_channels=True` shares one coefficient per offset and one bias across explicit channels.

These are software model choices; they are not part of the original two-dimensional symmetric circuit definition.

## Start from a classical template

```python
learnable = DifferentiableCellularNetwork.from_template(
    template,
    radius=1,
    steps=20,
)
```

Only scalar one-dimensional templates are accepted by this constructor. After training, scalar or single-channel models can be converted back with `to_template()`.

## Scientific lineage

Trajectory-based learning of CNN parameters predates modern automatic differentiation; {ref}`schuler-et-al-1992` formulates a trajectory error functional and derives parameter gradients with calculus of variations. CELNN does not claim to implement that historical algorithm. The connection is conceptual: both optimize parameters through the system’s time evolution, while this API relies on PyTorch autograd through CELNN’s shared numerical steps.
