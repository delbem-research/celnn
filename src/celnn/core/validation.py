"""Validation helpers used by core modules."""

from __future__ import annotations

import numpy as np
from numpy.typing import DTypeLike

from .exceptions import ShapeMismatchError, TemplateValidationError


def coerce_ndarray(
    value: object, *, dtype: DTypeLike | None = None, name: str = "value"
) -> np.ndarray:
    """Convert a value to a NumPy array."""
    try:
        return np.asarray(value, dtype=dtype)
    except Exception as exc:  # pragma: no cover - defensive path
        raise ShapeMismatchError(
            f"Could not convert {name} to a NumPy array."
        ) from exc


def validate_initial_state(
    initial_state: np.ndarray, state_shape: tuple[int, ...]
) -> None:
    """Validate an explicit initial state."""
    if tuple(initial_state.shape) != tuple(state_shape):
        raise ShapeMismatchError(
            "Initial state shape mismatch. "
            f"Expected {state_shape}, got {tuple(initial_state.shape)}."
        )


def ensure_broadcastable(
    value: object, state_shape: tuple[int, ...], name: str
) -> np.ndarray:
    """Validate broadcasting to a target shape."""
    array = np.asarray(value, dtype=float)
    try:
        return np.broadcast_to(array, state_shape).astype(float, copy=False)
    except ValueError as exc:
        raise ShapeMismatchError(
            f"{name} is not broadcastable to state shape {state_shape}. "
            f"Got shape {array.shape or 'scalar'}."
        ) from exc


def validate_template_shapes(
    feedback: np.ndarray, control: np.ndarray, state_ndim: int
) -> None:
    """Ensure feedback and control templates are mutually compatible."""
    if feedback.ndim != control.ndim:
        raise TemplateValidationError(
            "Feedback and control templates must have the same "
            "number of dimensions. "
            f"Got {feedback.ndim} and {control.ndim}."
        )
    if feedback.ndim != state_ndim:
        raise TemplateValidationError(
            "Template dimensionality must match the network dimensionality. "
            f"Got template ndim={feedback.ndim} and state ndim={state_ndim}."
        )
    if feedback.shape != control.shape:
        raise TemplateValidationError(
            "Feedback and control templates must share the same shape. "
            f"Got {feedback.shape} and {control.shape}."
        )
    if feedback.ndim == 0:
        raise TemplateValidationError(
            "Templates must be at least one-dimensional."
        )
    if any(size % 2 == 0 for size in feedback.shape):
        raise TemplateValidationError(
            "Templates must have odd extents along every axis "
            "so they have a center cell. "
            f"Got shape {feedback.shape}."
        )
