"""Continuous-time cellular dynamics helpers."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..backends import ArrayBackend


def local_feedback(
    state: Any,
    feedback: Any,
    activation: Callable[[Any], Any],
    backend: ArrayBackend,
    boundary: str,
    boundary_value: float,
) -> Any:
    """Compute the feedback contribution ``A * y(x)``."""
    output = activation(state)
    return backend.aggregate_local(
        output, feedback, mode=boundary, cval=boundary_value
    )


def local_control(
    input_array: Any,
    control: Any,
    backend: ArrayBackend,
    boundary: str,
    boundary_value: float,
) -> Any:
    """Compute the control contribution ``B * u``."""
    return backend.aggregate_local(
        input_array, control, mode=boundary, cval=boundary_value
    )


def local_drive(
    state: Any,
    input_array: Any,
    feedback: Any,
    control: Any,
    bias: Any,
    activation: Callable[[Any], Any],
    backend: ArrayBackend,
    boundary: str,
    boundary_value: float,
) -> Any:
    """Compute the non-decay part of the dynamics."""
    feedback_term = local_feedback(
        state, feedback, activation, backend, boundary, boundary_value
    )
    control_term = local_control(
        input_array, control, backend, boundary, boundary_value
    )
    return feedback_term + control_term + bias


def derivative(
    state: Any,
    input_array: Any,
    feedback: Any,
    control: Any,
    bias: Any,
    activation: Callable[[Any], Any],
    backend: ArrayBackend,
    boundary: str,
    boundary_value: float,
) -> Any:
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
