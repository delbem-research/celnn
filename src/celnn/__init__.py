"""Top-level package for celnn."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Any

from .activations import (
    identity,
    piecewise_linear,
    relu_activation,
    saturated_linear,
    sigmoid_activation,
    sign_activation,
    tanh_activation,
)
from .core.network import CellularNetwork
from .core.result import SimulationResult
from .core.simulation import SimulationConfig
from .core.templates import Template
from .templates.registry import TemplateRegistry

if TYPE_CHECKING:
    from .associative import (
        AssociativeMemoryState,
        DeltaHebbianMemory,
        DeltaHebbianRule,
    )
    from .associative_field import (
        AssociativeFieldState,
        NormalizedDeltaHebbianField,
    )
    from .differentiable import DifferentiableCellularNetwork
    from .plasticity import (
        HebbianRule,
        OjaRule,
        PlasticLinear,
        Plasticity,
        PlasticityRule,
        PlasticityState,
    )


def __getattr__(name: str) -> Any:
    """Expose optional APIs lazily so their dependencies stay optional."""
    if name == "DifferentiableCellularNetwork":
        from .differentiable import DifferentiableCellularNetwork

        return DifferentiableCellularNetwork
    if name in {
        "AssociativeFieldState",
        "AssociativeMemoryState",
        "DeltaHebbianMemory",
        "DeltaHebbianRule",
        "HebbianRule",
        "OjaRule",
        "PlasticLinear",
        "Plasticity",
        "PlasticityRule",
        "PlasticityState",
        "NormalizedDeltaHebbianField",
    }:
        from . import associative, associative_field, plasticity

        if hasattr(associative, name):
            module = associative
        elif hasattr(associative_field, name):
            module = associative_field
        else:
            module = plasticity
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "CellularNetwork",
    "AssociativeFieldState",
    "AssociativeMemoryState",
    "DeltaHebbianMemory",
    "DeltaHebbianRule",
    "DifferentiableCellularNetwork",
    "HebbianRule",
    "NormalizedDeltaHebbianField",
    "OjaRule",
    "PlasticLinear",
    "Plasticity",
    "PlasticityRule",
    "PlasticityState",
    "SimulationConfig",
    "SimulationResult",
    "Template",
    "TemplateRegistry",
    "identity",
    "piecewise_linear",
    "relu_activation",
    "saturated_linear",
    "sign_activation",
    "sigmoid_activation",
    "tanh_activation",
]

try:
    __version__ = version("celnn")
except PackageNotFoundError:  # pragma: no cover - source tree without install
    __version__ = "0+unknown"
