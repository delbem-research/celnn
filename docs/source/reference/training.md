# Training API

`celnn.training` deliberately exposes a small public surface. DEAP remains an
optional runtime dependency and is imported only when GA execution is requested.

```{py:currentmodule} celnn.training
```

```{autoclass} celnn.training.TrainingSample
:members:
```

```{autoclass} celnn.training.TrainingDataset
:members:
```

```{autodata} celnn.training.LossFn
```

```{autodata} celnn.training.NetworkFactory
```

```{autofunction} celnn.training.mse_loss
```

```{autofunction} celnn.training.l2_penalty
```

```{autofunction} celnn.training.evaluate_template
```

```{autoclass} celnn.training.GAConfig
:members:
```

```{autoclass} celnn.training.GATrainer
:members:
```

```{autoclass} celnn.training.GATrainingResult
:members:
```
