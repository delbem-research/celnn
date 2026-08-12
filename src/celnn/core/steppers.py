"""Time-integration formulas, shared by every execution path.

These are the only places the library decides how a cellular state advances in
time. Both the NumPy simulator and the differentiable torch network integrate
through them, so a trajectory computed for analysis and a trajectory
backpropagated through are the same trajectory by construction.

The formulas are pure arithmetic and therefore work on any array type: NumPy,
CuPy or torch tensors, with gradients intact.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

__all__ = ["STEPPERS", "euler_step", "semi_implicit_euler_step"]


def euler_step(state: Any, dt: float, derivative: Any) -> Any:
    """Explicit Euler: ``x <- x + dt * f(x)``."""
    return state + dt * derivative


def semi_implicit_euler_step(state: Any, dt: float, drive: Any) -> Any:
    """Semi-implicit Euler, treating the leak term ``-x`` implicitly.

    Solving ``x' = x + dt * (-x' + drive)`` for ``x'`` gives the expression
    below. ``drive`` is the non-decay part of the dynamics, ``A * y(x) + B * u
    + z``, not the full derivative.
    """
    return (state + dt * drive) / (1.0 + dt)


STEPPERS: dict[str, Callable[[Any, float, Any], Any]] = {
    "euler": euler_step,
    "semi_implicit_euler": semi_implicit_euler_step,
}
