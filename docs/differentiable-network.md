# Differentiable Cellular Network

`DifferentiableCellularNetwork` is the PyTorch form of the same continuous-time
CelNN implemented by `CellularNetwork`. Its feedback, control, and bias
templates can be learned by backpropagation through multiple integration
steps.

Install the optional dependency:

```bash
pip install "celnn[torch]"
```

The class is available directly from the package:

```python
import torch

from celnn import DifferentiableCellularNetwork

net = DifferentiableCellularNetwork(
    radius=2,
    channels=16,
    dt=0.1,
    steps=10,
)

field = torch.randn(8, 64, 16)
result = net(field)
loss = result.square().mean()
loss.backward()
```

The parameters work with ordinary PyTorch optimizers. For example, fitting a
network to a target trajectory follows the usual training loop:

```python
target = torch.zeros_like(field)
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

for _ in range(100):
    optimizer.zero_grad()
    loss = torch.nn.functional.mse_loss(net(field), target)
    loss.backward()
    optimizer.step()
```

## Reusing classical templates

A one-dimensional `Template` can initialize the differentiable network. The
learned scalar or single-channel parameters can then return to the reference
simulator:

```python
from celnn import CellularNetwork, DifferentiableCellularNetwork
from celnn.core.templates import Template

template = Template(
    name="example",
    feedback=[0.2, 1.0, 0.2],
    control=[0.1, 0.8, 0.1],
    bias=0.0,
)

learnable = DifferentiableCellularNetwork.from_template(
    template,
    radius=1,
    steps=20,
)
recovered = learnable.to_template(name="learned")
reference = CellularNetwork.from_template(recovered, input=[0.0] * 32)
```

Passing `trainable=False` registers the templates as buffers instead of
parameters. The supported integration methods are `euler` and
`semi_implicit_euler`.

## Scalar and channelled contracts

When `channels` is omitted, the input follows the scalar reference contract;
the final dimension is spatial. When `channels` is supplied, input has shape
`(..., length, channels)` and templates have shape `(2 * radius + 1,
channels)`.

The channelled templates are diagonal: channels evolve independently and do
not communicate inside the CelNN dynamics. Cross-channel mixing must be an
explicit model component outside this class.

For identical scalar templates, inputs, initial states, integration settings,
and boundaries, the differentiable path is tested against the reference
simulator at `rtol=1e-9` and `atol=1e-9` in `float64`. Both paths call the same
canonical ODE and time-step formulas; the PyTorch form adds parameter ownership
and graph-preserving iteration rather than a second mathematical definition.

PyTorch remains optional. A plain `import celnn` does not import PyTorch; the
dependency is loaded only when `DifferentiableCellularNetwork` is requested.
