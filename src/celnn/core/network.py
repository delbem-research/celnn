"""Main user-facing CellularNetwork class."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from typing import Any

import numpy as np

from ..backends import get_backend
from .activations import activation_name, resolve_activation
from .boundary import normalize_boundary_mode
from .dynamics import derivative as compute_derivative
from .dynamics import local_drive
from .exceptions import CelNNError
from .result import SimulationResult
from .simulation import SimulationConfig
from .solvers import solve
from .steppers import euler_step
from .templates import Template
from .topology import RegularGridTopology
from .validation import (
    coerce_ndarray,
    ensure_broadcastable,
    validate_initial_state,
)

_ALLOWED_SERIALIZED_KEYS = {
    "input",
    "initial_state",
    "current_state",
    "feedback",
    "control",
    "bias",
    "activation",
    "boundary",
    "boundary_value",
    "dtype",
    "metadata",
}
_SUPPORTED_DTYPES = frozenset(
    {np.dtype(np.float32), np.dtype(np.float64)}
)


def _normalize_dtype(value: Any | None) -> np.dtype[Any]:
    try:
        dtype = np.dtype(np.float64 if value is None else value)
    except (TypeError, ValueError) as exc:
        raise CelNNError(f"Unsupported dtype {value!r}.") from exc
    if dtype not in _SUPPORTED_DTYPES:
        raise CelNNError(
            "dtype must be float32 or float64, "
            f"got {dtype.name!r}."
        )
    return dtype


class CellularNetwork:
    """Continuous-time Cellular Neural Network over a regular grid."""

    def __init__(
        self,
        input: Any,
        initial_state: Any | None = None,
        feedback: Any | None = None,
        control: Any | None = None,
        bias: Any = 0.0,
        activation: str | Any = "piecewise_linear",
        boundary: str = "constant",
        boundary_value: float = 0.0,
        dtype: Any | None = None,
        device: str = "cpu",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.dtype = _normalize_dtype(dtype)
        self.input = coerce_ndarray(input, dtype=self.dtype, name="input")
        self.state_shape = tuple(self.input.shape)

        self.topology = RegularGridTopology(
            shape=self.state_shape,
            boundary=boundary,
            boundary_value=boundary_value,
        )
        self.boundary = normalize_boundary_mode(boundary)
        self.boundary_value = float(boundary_value)
        self.metadata = deepcopy(metadata) if metadata is not None else {}
        if not isinstance(device, str):
            raise CelNNError("device must be a string.")
        self.device = device.lower().strip()
        self.backend = get_backend(self.device)

        self.feedback = self._resolve_template_array(
            value=feedback,
            default=self.topology.identity_template,
            name="feedback",
        )
        self.control = self._resolve_template_array(
            value=control,
            default=lambda: np.zeros_like(self.feedback),
            name="control",
        )
        self.topology.validate_template(self.feedback, self.control)
        self.bias = ensure_broadcastable(
            bias, self.state_shape, "bias"
        ).astype(self.dtype, copy=False)
        self.activation = activation
        self._activation_fn = resolve_activation(activation)

        if initial_state is None:
            self._initial_state = np.zeros(self.state_shape, dtype=self.dtype)
        else:
            initial_array = coerce_ndarray(
                initial_state, dtype=self.dtype, name="initial_state"
            )
            validate_initial_state(initial_array, self.state_shape)
            self._initial_state = initial_array

        self.state = self._initial_state.copy()
        self._last_solver = "euler"

    def _resolve_template_array(
        self,
        value: Any | None,
        default: Callable[[], Any],
        name: str,
    ) -> np.ndarray:
        if value is None:
            return np.asarray(default(), dtype=self.dtype)
        return coerce_ndarray(value, dtype=self.dtype, name=name)

    @classmethod
    def from_template(
        cls,
        template: Template,
        input: Any,
        *,
        initial_state: Any | None = None,
        activation: str | Any = "piecewise_linear",
        boundary: str = "constant",
        boundary_value: float = 0.0,
        dtype: Any | None = None,
        device: str = "cpu",
        metadata: dict[str, Any] | None = None,
    ) -> "CellularNetwork":
        """Construct a network from a reusable template."""
        template.validate()
        chosen_initial_state = (
            initial_state
            if initial_state is not None
            else template.initial_state
        )
        combined_metadata = deepcopy(template.metadata)
        if metadata:
            combined_metadata.update(metadata)
        combined_metadata.setdefault("template_name", template.name)
        return cls(
            input=input,
            initial_state=chosen_initial_state,
            feedback=template.feedback,
            control=template.control,
            bias=template.bias,
            activation=activation,
            boundary=boundary,
            boundary_value=boundary_value,
            dtype=dtype,
            device=device,
            metadata=combined_metadata,
        )

    def output(self, state: np.ndarray | None = None) -> np.ndarray:
        """Return the current output field ``y(x)``."""
        chosen_state = (
            self.state
            if state is None
            else np.asarray(state, dtype=self.dtype)
        )
        return np.asarray(self._activation_fn(chosen_state), dtype=self.dtype)

    def drive(self, state: np.ndarray) -> np.ndarray:
        """Return ``A*y(x) + B*u + z``."""
        chosen_state = np.asarray(state, dtype=self.dtype)
        return np.asarray(
            local_drive(
                state=chosen_state,
                input_array=self.input,
                feedback=self.feedback,
                control=self.control,
                bias=self.bias,
                activation=self._activation_fn,
                backend=self.backend,
                boundary=self.boundary,
                boundary_value=self.boundary_value,
            ),
            dtype=self.dtype,
        )

    def derivative(self, state: np.ndarray) -> np.ndarray:
        """Return ``dx/dt`` evaluated at a given state."""
        chosen_state = np.asarray(state, dtype=self.dtype)
        return np.asarray(
            compute_derivative(
                state=chosen_state,
                input_array=self.input,
                feedback=self.feedback,
                control=self.control,
                bias=self.bias,
                activation=self._activation_fn,
                backend=self.backend,
                boundary=self.boundary,
                boundary_value=self.boundary_value,
            ),
            dtype=self.dtype,
        )

    def step(self, dt: float) -> np.ndarray:
        """Advance one explicit Euler step and return the new state."""
        if dt <= 0:
            raise CelNNError(f"dt must be positive, got {dt}.")
        self.state = np.asarray(
            euler_step(self.state, float(dt), self.derivative(self.state)),
            dtype=self.dtype,
        )
        return self.state.copy()

    def run(
        self,
        config: SimulationConfig | None = None,
    ) -> SimulationResult:
        """Run the simulation and return a result object."""
        chosen_config = config if config is not None else SimulationConfig()
        return solve(self, chosen_config)

    def reset(self, initial_state: Any | None = None) -> None:
        """Reset the internal state."""
        if initial_state is None:
            self.state = self._initial_state.copy()
            return
        initial_array = coerce_ndarray(
            initial_state, dtype=self.dtype, name="initial_state"
        )
        validate_initial_state(initial_array, self.state_shape)
        self._initial_state = initial_array.copy()
        self.state = initial_array.copy()

    def to_dict(self) -> dict[str, Any]:
        """Serialize the semantic network configuration and current state."""
        activation_key = activation_name(self.activation)
        if activation_key is None:
            raise CelNNError(
                "Custom activation callables cannot be serialized "
                "automatically. Use a named built-in activation or "
                "provide your own serialization layer."
            )
        return {
            "input": self.input.tolist(),
            "initial_state": self._initial_state.tolist(),
            "current_state": self.state.tolist(),
            "feedback": self.feedback.tolist(),
            "control": self.control.tolist(),
            "bias": np.asarray(self.bias).tolist(),
            "activation": activation_key,
            "boundary": self.boundary,
            "boundary_value": self.boundary_value,
            "dtype": self.dtype.name,
            "metadata": deepcopy(self.metadata),
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        *,
        device: str = "cpu",
    ) -> "CellularNetwork":
        """Restore a validated network from semantic serialized state."""
        if not isinstance(data, dict):
            raise CelNNError("CellularNetwork data must be a dictionary.")
        unknown = set(data) - _ALLOWED_SERIALIZED_KEYS
        if unknown:
            names = ", ".join(sorted(unknown))
            raise CelNNError(f"Unknown CellularNetwork fields: {names}.")
        if "input" not in data:
            raise CelNNError("CellularNetwork data is missing 'input'.")
        for name in ("activation", "boundary", "dtype"):
            if name in data and not isinstance(data[name], str):
                raise CelNNError(f"{name} must be a string.")
        metadata = data.get("metadata", {})
        if not isinstance(metadata, dict):
            raise CelNNError("metadata must be a dictionary.")

        boundary_value = data.get("boundary_value", 0.0)
        if isinstance(boundary_value, bool) or not isinstance(
            boundary_value, (int, float)
        ):
            raise CelNNError("boundary_value must be a real number.")

        network = cls(
            input=data["input"],
            initial_state=data.get("initial_state"),
            feedback=data.get("feedback"),
            control=data.get("control"),
            bias=data.get("bias", 0.0),
            activation=data.get("activation", "piecewise_linear"),
            boundary=data.get("boundary", "constant"),
            boundary_value=float(boundary_value),
            dtype=data.get("dtype"),
            device=device,
            metadata=deepcopy(metadata),
        )
        current_state = data.get("current_state")
        if current_state is not None:
            current_array = np.asarray(current_state, dtype=network.dtype)
            validate_initial_state(current_array, network.state_shape)
            network.state = current_array
        return network
