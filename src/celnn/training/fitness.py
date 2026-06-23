"""Fitness evaluation utilities for genetic-algorithm template training.

This module provides the building blocks used by
:mod:`celnn.training.ga` to score candidate templates against a
:class:`celnn.training.dataset.TrainingDataset`.

The fitness function is intentionally simple: it averages the per-sample
mean squared error between the network output produced by a candidate
template and the target output. Optional L2 regularization can be added
to favor smaller template coefficients, which often improves numerical
stability of the underlying CelNN dynamics.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

from ..core.exceptions import CelNNError
from ..core.network import CellularNetwork
from ..core.simulation import SimulationConfig

if TYPE_CHECKING:
    from .dataset import TrainingDataset, TrainingSample


LossFn = Callable[[np.ndarray, np.ndarray], float]
"""Signature for a per-sample loss function: (target, output) -> float."""


NetworkFactory = Callable[["TrainingSample"], CellularNetwork]
"""Signature for building a network from a single training sample."""


def mse_loss(target: np.ndarray, output: np.ndarray) -> float:
    """Return the mean squared error between ``output`` and ``target``."""
    target_array = np.asarray(target, dtype=float)
    output_array = np.asarray(output, dtype=float)
    if target_array.shape != output_array.shape:
        raise CelNNError(
            "Target and output must share the same shape. "
            f"Got target={target_array.shape} and output={output_array.shape}."
        )
    return float(np.mean((output_array - target_array) ** 2))


def l2_penalty(
    feedback: np.ndarray, control: np.ndarray, bias: np.ndarray
) -> float:
    """Return the L2 norm of the (feedback, control, bias) coefficients."""
    return float(
        np.sum(np.asarray(feedback, dtype=float) ** 2)
        + np.sum(np.asarray(control, dtype=float) ** 2)
        + np.sum(np.asarray(bias, dtype=float) ** 2)
    )


def evaluate_template(
    network_factory: NetworkFactory,
    config: SimulationConfig,
    dataset: "TrainingDataset",
    *,
    loss_fn: LossFn = mse_loss,
    regularization: float = 0.0,
    penalty_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], float]
    | None = l2_penalty,
    max_loss: float = 1e12,
) -> float:
    """Evaluate a candidate template across all dataset samples.

    Parameters
    ----------
    network_factory
        Callable that, given a sample, returns a configured
        :class:`CellularNetwork` ready to be simulated.
    config
        Simulation configuration used to advance the network.
    dataset
        Training dataset providing (input, target) pairs.
    loss_fn
        Per-sample loss function. Defaults to mean squared error.
    regularization
        Multiplier applied to the L2 penalty. ``0.0`` disables
        regularization.
    penalty_fn
        Optional penalty function over the feedback, control, and bias
        arrays. When ``None`` no penalty is applied.
    max_loss
        Upper bound returned when a simulation fails or produces non
        finite values. The bound keeps the genetic algorithm from being
        dominated by catastrophic candidates.

    Returns
    -------
    float
        The aggregated fitness value (lower is better).
    """
    total = 0.0
    count = 0
    last_network: CellularNetwork | None = None
    for sample in dataset.samples:
        try:
            network = network_factory(sample)
            result = network.run(config)
            output = np.asarray(result.output, dtype=float)
            sample_loss = loss_fn(sample.target, output)
        except CelNNError:
            return max_loss
        except FloatingPointError:
            return max_loss
        if not np.isfinite(sample_loss):
            return max_loss
        total += float(sample_loss)
        count += 1
        last_network = network

    if count == 0:
        return max_loss

    mean_loss = total / count
    if (
        regularization > 0.0
        and penalty_fn is not None
        and last_network is not None
    ):
        mean_loss = mean_loss + regularization * penalty_fn(
            np.asarray(last_network.feedback, dtype=float),
            np.asarray(last_network.control, dtype=float),
            np.asarray(last_network.bias, dtype=float),
        )
    return float(mean_loss)
