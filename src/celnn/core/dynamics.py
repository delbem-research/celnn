"""Continuous-time cellular dynamics helpers."""

from __future__ import annotations

import numpy as np

from ..backends import ArrayBackend


def local_feedback(
    state: np.ndarray,
    feedback: np.ndarray,
    activation,
    backend: ArrayBackend,
    boundary: str,
    boundary_value: float,
) -> np.ndarray:
    """Compute the feedback contribution ``A * y(x)``."""
    output = activation(state)
    return backend.aggregate_local(
        output, feedback, mode=boundary, cval=boundary_value
    )


def local_control(
    input_array: np.ndarray,
    control: np.ndarray,
    backend: ArrayBackend,
    boundary: str,
    boundary_value: float,
) -> np.ndarray:
    """Compute the control contribution ``B * u``."""
    return backend.aggregate_local(
        input_array, control, mode=boundary, cval=boundary_value
    )


def local_drive(
    state: np.ndarray,
    input_array: np.ndarray,
    feedback: np.ndarray,
    control: np.ndarray,
    bias: np.ndarray,
    activation,
    backend: ArrayBackend,
    boundary: str,
    boundary_value: float,
) -> np.ndarray:
    """Compute the non-decay part of the dynamics."""
    feedback_term = local_feedback(
        state, feedback, activation, backend, boundary, boundary_value
    )
    control_term = local_control(
        input_array, control, backend, boundary, boundary_value
    )
    return feedback_term + control_term + bias


def derivative(
    state: np.ndarray,
    input_array: np.ndarray,
    feedback: np.ndarray,
    control: np.ndarray,
    bias: np.ndarray,
    activation,
    backend: ArrayBackend,
    boundary: str,
    boundary_value: float,
) -> np.ndarray:
    """Compute ``dx/dt = -x + A*y(x) + B*u + z``."""
    drive = local_drive(
        state=state,
        input_array=input_array,
        feedback=feedback,
        control=control,
        bias=bias,
        activation=activation,
        backend=backend,
        boundary=boundary,
        boundary_value=boundary_value,
    )
    return -state + drive
