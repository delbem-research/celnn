# Train a template with a genetic algorithm

Install the DEAP-backed training capability:

```bash
python -m pip install "celnn[ga]"
```

CELNN’s GA trainer searches the feedback coefficients, control coefficients, and one scalar bias of a template. The template shape defines the search representation; a dataset and simulation configuration define how candidates are evaluated.

The use of genetic search for CNN template learning has historical precedent in {ref}`kozek-roska-chua-1993`. CELNN’s implementation is its own software contract; it should not be assumed to reproduce every encoding or hyperparameter choice from that paper.

## Prepare a dataset

```python
import numpy as np
from celnn.training import TrainingDataset

inputs = [
    np.linspace(-1.0, 1.0, 32),
    np.sin(np.linspace(0.0, 2.0 * np.pi, 32)),
]
targets = [np.zeros(32), np.zeros(32)]
dataset = TrainingDataset.from_pairs(inputs, targets)
```

## Define the search shape and simulation

```python
from celnn import SimulationConfig, Template
from celnn.training import GAConfig, GATrainer

seed_template = Template(
    name="search_shape",
    feedback=[0.0, 1.0, 0.0],
    control=[0.0, 1.0, 0.0],
    bias=0.0,
)

trainer = GATrainer(
    template=seed_template,
    dataset=dataset,
    config=SimulationConfig(t_end=1.0, dt=0.05),
    ga_config=GAConfig(
        pop_size=30,
        n_generations=20,
        bounds=(-2.0, 2.0),
        seed_template=seed_template,
    ),
    seed=42,
)
```

A fixed integer `seed` makes the trainer’s random sources deterministic for reproducible experiments.

## Run and inspect the result

```python
training = trainer.run()
print(training.best_fitness)
best = training.best_template
```

{py:class}`celnn.training.GATrainingResult` also records fitness histories, evaluation counts, population size, configuration, and the seed.

## Keep the search objective honest

A low training loss means the candidate performed well under the chosen dataset, simulation configuration, loss, and regularization. It is not automatically evidence of stability, generalization, or robustness. Evaluate those properties separately on held-out conditions or with domain-specific invariants.
