"""Training dataset container for the genetic-algorithm trainer.

The :class:`TrainingDataset` is a thin convenience wrapper over a list of
:class:`TrainingSample` instances. The wrapper performs basic shape
validation up front and provides a few class-method constructors for the
common "list of inputs + list of targets" pattern.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..core.exceptions import ShapeMismatchError


@dataclass(slots=True)
class TrainingSample:
    """A single (input, target) pair used for fitness evaluation."""

    input: np.ndarray
    target: np.ndarray
    initial_state: np.ndarray | None = None
    weight: float = 1.0

    def __post_init__(self) -> None:
        self.input = np.asarray(self.input, dtype=float)
        self.target = np.asarray(self.target, dtype=float)
        if self.initial_state is not None:
            self.initial_state = np.asarray(self.initial_state, dtype=float)
        if self.input.shape != self.target.shape:
            raise ShapeMismatchError(
                "Training sample input and target must share the same shape. "
                f"Got input={self.input.shape} and target={self.target.shape}."
            )


@dataclass(slots=True)
class TrainingDataset:
    """A collection of :class:`TrainingSample` objects."""

    samples: list[TrainingSample]

    def __post_init__(self) -> None:
        if not self.samples:
            raise ValueError(
                "TrainingDataset must contain at least one sample."
            )
        reference_shape = self.samples[0].input.shape
        for index, sample in enumerate(self.samples):
            if sample.input.shape != reference_shape:
                raise ShapeMismatchError(
                    "All training samples must share the same input shape. "
                    f"Sample 0 has shape {reference_shape} while sample "
                    f"{index} has shape {sample.input.shape}."
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self) -> Iterable[TrainingSample]:
        return iter(self.samples)

    @classmethod
    def from_pairs(
        cls,
        inputs: Sequence[Any] | np.ndarray,
        targets: Sequence[Any] | np.ndarray,
        initial_states: Sequence[Any] | np.ndarray | None = None,
        weights: Sequence[float] | None = None,
    ) -> "TrainingDataset":
        """Build a dataset from parallel lists of inputs and targets.

        Parameters
        ----------
        inputs
            Sequence of input arrays.
        targets
            Sequence of target arrays, with the same length as ``inputs``.
        initial_states
            Optional sequence of initial states matching ``inputs``.
        weights
            Optional per-sample scalar weights used to scale the
            contribution of each sample to the average loss.
        """
        if len(inputs) != len(targets):
            raise ValueError(
                "inputs and targets must have the same length. "
                f"Got {len(inputs)} and {len(targets)}."
            )
        if initial_states is not None and len(initial_states) != len(inputs):
            raise ValueError(
                "initial_states must have the same length as inputs. "
                f"Got {len(initial_states)} and {len(inputs)}."
            )
        if weights is not None and len(weights) != len(inputs):
            raise ValueError(
                "weights must have the same length as inputs. "
                f"Got {len(weights)} and {len(inputs)}."
            )
        samples: list[TrainingSample] = []
        for index, (input_array, target_array) in enumerate(
            zip(inputs, targets, strict=True)
        ):
            initial = (
                None
                if initial_states is None
                else initial_states[index]
            )
            weight = 1.0 if weights is None else float(weights[index])
            samples.append(
                TrainingSample(
                    input=input_array,
                    target=target_array,
                    initial_state=initial,
                    weight=weight,
                )
            )
        return cls(samples)
