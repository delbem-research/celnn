# Examples

This document mirrors the runnable scripts in `examples/`.

## 1. Image edge detection

```python
from celnn import CellularNetwork, SimulationConfig
from celnn.activations import piecewise_linear
from celnn.domains.image import load_grayscale, save_grayscale
from celnn.templates.image_processing import EDGE_DETECTION

u = load_grayscale("input.png")

net = CellularNetwork(
    input=u,
    feedback=EDGE_DETECTION.feedback,
    control=EDGE_DETECTION.control,
    bias=EDGE_DETECTION.bias,
    activation=piecewise_linear,
    boundary="reflect",
)

result = net.run(SimulationConfig(t_start=0.0, t_end=10.0, dt=0.1))
save_grayscale(result.output, "edge.png")
```

## 2. Generic 1D signal filtering

```python
import numpy as np
from celnn import CellularNetwork, SimulationConfig
from celnn.activations import tanh_activation

signal = np.sin(np.linspace(0, 8 * np.pi, 512))

net = CellularNetwork(
    input=signal,
    feedback=np.array([0.2, 1.0, 0.2]),
    control=np.array([0.1, 0.8, 0.1]),
    bias=0.0,
    activation=tanh_activation,
    boundary="reflect",
)

result = net.run(
    SimulationConfig(
        t_start=0.0,
        t_end=5.0,
        dt=0.05,
        return_trajectory=True,
    )
)
```

## 3. Custom pattern formation

```python
import numpy as np
from celnn import CellularNetwork, SimulationConfig
from celnn.activations import saturated_linear

initial = np.random.uniform(-1, 1, size=(64, 64))
external_input = np.zeros((64, 64))

A = np.array([
    [0.05, 0.20, 0.05],
    [0.20, 1.00, 0.20],
    [0.05, 0.20, 0.05],
])

B = np.zeros((3, 3))

net = CellularNetwork(
    input=external_input,
    initial_state=initial,
    feedback=A,
    control=B,
    bias=-0.1,
    activation=saturated_linear,
    boundary="wrap",
)

result = net.run(SimulationConfig(t_start=0.0, t_end=20.0, dt=0.1))
```

## 4. Template registry

```python
from celnn.templates import Template, TemplateRegistry

registry = TemplateRegistry()
registry.register(
    Template(
        name="custom_edge_detector",
        feedback=[[0, 0, 0], [0, 1, 0], [0, 0, 0]],
        control=[[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]],
        bias=-1.0,
        description="Custom edge-like detector",
        tags=["image", "edge", "demo"],
    )
)
```

## 5. Direct template usage

```python
from celnn import CellularNetwork, SimulationConfig
from celnn.templates.image_processing import EDGE_DETECTION

net = CellularNetwork.from_template(
    template=EDGE_DETECTION,
    input=my_array,
    activation="piecewise_linear",
    boundary="reflect",
)

result = net.run(SimulationConfig(t_end=5.0, dt=0.05))
```

## 6. Serialization

```python
from celnn.templates import Template

template = Template(
    name="my_template",
    feedback=[[0, 1, 0], [1, 2, 1], [0, 1, 0]],
    control=[[0, 0, 0], [0, 1, 0], [0, 0, 0]],
    bias=0.0,
)

data = template.to_dict()
restored = Template.from_dict(data)
```

## 7. Genetic-algorithm template training (DEAP)

```python
import numpy as np
from celnn import SimulationConfig
from celnn.templates import Template
from celnn.training import GAConfig, GATrainer, TrainingDataset

template = Template(
    name="smoothing_seed",
    feedback=[0.05, 0.1, 0.05],
    control=[0.0, 1.0, 0.0],
    bias=0.0,
)

inputs = [np.linspace(-1.0, 1.0, 32), np.sin(np.linspace(0.0, 2.0 * np.pi, 32))]
targets = [np.convolve(u, np.ones(5) / 5.0, mode="same") for u in inputs]
dataset = TrainingDataset.from_pairs(inputs, targets)

trainer = GATrainer(
    template=template,
    dataset=dataset,
    config=SimulationConfig(t_end=5.0, dt=0.1, solver="euler"),
    ga_config=GAConfig(
        pop_size=20,
        n_generations=10,
        bounds=(-1.0, 1.0),
        bias_bounds=(-0.5, 0.5),
        seed_template=template,
    ),
    activation="identity",
    boundary="reflect",
    seed=42,
)

result = trainer.run()
print(result.best_fitness)
print(result.best_template.feedback)
```
