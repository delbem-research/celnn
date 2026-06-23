"""Genetic-algorithm-based training for CelNN templates.

The :mod:`celnn.training` subpackage bundles a small genetic-algorithm
implementation, powered by the
`DEAP <https://deap.readthedocs.io/en/master/>`_ library, that searches
for the ``(A, B, z)`` coefficients of a :class:`celnn.core.templates.Template`
that best fit a dataset of (input, target) pairs.

DEAP is an optional dependency. Install it with::

    pip install celnn[ga]
"""

from __future__ import annotations

from .dataset import TrainingDataset, TrainingSample
from .fitness import (
    LossFn,
    NetworkFactory,
    evaluate_template,
    l2_penalty,
    mse_loss,
)
from .ga import GAConfig, GATrainer, GATrainingResult

__all__ = [
    "GAConfig",
    "GATrainer",
    "GATrainingResult",
    "LossFn",
    "NetworkFactory",
    "TrainingDataset",
    "TrainingSample",
    "evaluate_template",
    "l2_penalty",
    "mse_loss",
]
