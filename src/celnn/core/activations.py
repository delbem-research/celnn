"""Built-in activation functions for cellular outputs.

Each activation is written once and evaluated by whichever array library owns
its input -- NumPy, CuPy or PyTorch. The expressions below deliberately stay
inside the intersection of those libraries: arithmetic, the builtin ``abs``,
and the handful of functions (``tanh``, ``exp``, ``sign``) that all three spell
identically. That keeps the differentiable path from needing a second, parallel
implementation that could drift away from this one.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .arrays import array_namespace, as_float_array
from .exceptions import CelNNError


def piecewise_linear(x: Any) -> Any:
    """Classic bounded piecewise-linear activation for many CelNN models."""
    x = as_float_array(x)
    return 0.5 * (abs(x + 1.0) - abs(x - 1.0))


def saturated_linear(x: Any) -> Any:
    """Saturating linear output in ``[-1, 1]``.

    Expressed through ``piecewise_linear`` because the two are the same
    function; clipping and the absolute-value form agree everywhere.
    """
    return piecewise_linear(x)


def identity(x: Any) -> Any:
    """Identity activation."""
    return as_float_array(x)


def tanh_activation(x: Any) -> Any:
    """Hyperbolic tangent activation."""
    x = as_float_array(x)
    return array_namespace(x).tanh(x)


def sigmoid_activation(x: Any) -> Any:
    """Logistic activation in ``[0, 1]``."""
    x = as_float_array(x)
    return 1.0 / (1.0 + array_namespace(x).exp(-x))


def sign_activation(x: Any) -> Any:
    """Sign activation."""
    x = as_float_array(x)
    return array_namespace(x).sign(x)


def relu_activation(x: Any) -> Any:
    """Rectified linear activation.

    Written as a masked product rather than ``maximum``: NumPy and PyTorch
    disagree on whether the second argument may be a Python scalar.
    """
    x = as_float_array(x)
    return x * (x > 0)


ACTIVATIONS: dict[str, Callable[[Any], Any]] = {
    "piecewise_linear": piecewise_linear,
    "saturated_linear": saturated_linear,
    "identity": identity,
    "tanh_activation": tanh_activation,
    "sigmoid_activation": sigmoid_activation,
    "sign_activation": sign_activation,
    "relu_activation": relu_activation,
}


def resolve_activation(
    name_or_callable: str | Callable[[Any], Any],
) -> Callable[[Any], Any]:
    """Resolve an activation specified by name or callable."""
    if callable(name_or_callable):
        return name_or_callable
    key = str(name_or_callable).strip()
    try:
        return ACTIVATIONS[key]
    except KeyError as exc:
        known = ", ".join(sorted(ACTIVATIONS))
        raise CelNNError(
            f"Unknown activation '{name_or_callable}'. "
            f"Known activations: {known}."
        ) from exc


def activation_name(
    name_or_callable: str | Callable[[Any], Any],
) -> str | None:
    """Return a stable activation name if available."""
    if isinstance(name_or_callable, str):
        return name_or_callable
    for name, func in ACTIVATIONS.items():
        if func is name_or_callable:
            return name
    return None


__all__ = [
    "ACTIVATIONS",
    "activation_name",
    "identity",
    "piecewise_linear",
    "relu_activation",
    "resolve_activation",
    "saturated_linear",
    "sigmoid_activation",
    "sign_activation",
    "tanh_activation",
]
