"""Differentiable Cellular Neural Network with learnable templates.

This module adds PyTorch parameter ownership and repeated integration while
reusing the reference CelNN dynamics, stencil backend, activations, and time
steppers.  PyTorch is imported only when this optional public API is requested.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

try:
    import torch
except ImportError as exc:  # pragma: no cover - environment branch
    raise ImportError(
        "DifferentiableCellularNetwork requires PyTorch. "
        "Install it with `pip install celnn[torch]`."
    ) from exc

from .backends.torch_backend import TorchBackend
from .core.activations import resolve_activation
from .core.boundary import normalize_boundary_mode
from .core.dynamics import derivative as compute_derivative
from .core.dynamics import local_drive
from .core.steppers import STEPPERS
from .core.templates import Template


class DifferentiableCellularNetwork(torch.nn.Module):
    """A one-dimensional CelNN whose templates are PyTorch tensors.

    Omitting ``channels`` gives the scalar reference contract. Supplying it,
    including ``channels=1``, adds an explicit final channel axis and diagonal
    per-channel templates. Channels do not mix inside these dynamics.
    """

    def __init__(
        self,
        radius: int,
        channels: int | None = None,
        activation: str | Callable[[Any], Any] = "piecewise_linear",
        boundary: str = "constant",
        boundary_value: float = 0.0,
        dt: float = 0.1,
        steps: int = 10,
        method: str = "euler",
        *,
        causal: bool = False,
        shared_channels: bool = False,
        trainable: bool = True,
    ) -> None:
        super().__init__()
        if (
            not isinstance(radius, int)
            or isinstance(radius, bool)
            or radius < 0
        ):
            raise ValueError("radius must be a non-negative integer.")
        if channels is not None and (
            not isinstance(channels, int)
            or isinstance(channels, bool)
            or channels < 1
        ):
            raise ValueError("channels must be a positive integer or None.")
        if not isinstance(steps, int) or isinstance(steps, bool) or steps < 0:
            raise ValueError("steps must be a non-negative integer.")
        if dt <= 0:
            raise ValueError("dt must be positive.")
        if method not in STEPPERS:
            known = ", ".join(sorted(STEPPERS))
            raise ValueError(
                f"Unknown integration method {method!r}. "
                f"Known methods: {known}."
            )

        if shared_channels and channels is None:
            raise ValueError(
                "shared_channels requires an explicit channel count."
            )

        span = radius + 1 if causal else 2 * radius + 1
        parameter_shape = (
            (span,)
            if channels is None
            else (span, 1 if shared_channels else channels)
        )
        bias_shape = (
            ()
            if channels is None
            else (1 if shared_channels else channels,)
        )

        self.radius = radius
        self.channels = channels
        self.activation = activation
        self.boundary = normalize_boundary_mode(boundary)
        self.boundary_value = float(boundary_value)
        self.dt = float(dt)
        self.steps = steps
        self.method = method
        self.causal = bool(causal)
        self.shared_channels = bool(shared_channels)
        self.trainable = bool(trainable)
        self._activation_fn = resolve_activation(activation)
        self.backend = TorchBackend(spatial_ndim=1, causal=self.causal)

        feedback = torch.zeros(parameter_shape)
        control = torch.zeros(parameter_shape)
        bias = torch.zeros(bias_shape)
        if self.trainable:
            self.feedback = torch.nn.Parameter(feedback)
            self.control = torch.nn.Parameter(control)
            self.bias = torch.nn.Parameter(bias)
        else:
            self.register_buffer("feedback", feedback)
            self.register_buffer("control", control)
            self.register_buffer("bias", bias)

    @classmethod
    def from_template(
        cls,
        template: Template,
        radius: int,
        channels: int | None = None,
        activation: str | Callable[[Any], Any] = "piecewise_linear",
        boundary: str = "constant",
        boundary_value: float = 0.0,
        dt: float = 0.1,
        steps: int = 10,
        method: str = "euler",
        *,
        trainable: bool = True,
    ) -> "DifferentiableCellularNetwork":
        """Construct a differentiable network from a scalar 1-D template."""
        template.validate()
        feedback = np.asarray(template.feedback, dtype=float)
        control = np.asarray(template.control, dtype=float)
        expected_span = 2 * radius + 1
        if feedback.ndim != 1 or control.ndim != 1:
            raise ValueError(
                "Differentiable templates must be one-dimensional."
            )
        if feedback.shape != (expected_span,) or control.shape != (
            expected_span,
        ):
            raise ValueError(
                "Template extent must match radius: expected "
                f"{expected_span} coefficients."
            )

        network = cls(
            radius=radius,
            channels=channels,
            activation=activation,
            boundary=boundary,
            boundary_value=boundary_value,
            dt=dt,
            steps=steps,
            method=method,
            trainable=trainable,
        )
        network.double()
        network._load_template(feedback, control, template.bias)
        return network

    def _load_template(self, feedback: Any, control: Any, bias: Any) -> None:
        """Copy scalar templates, broadcasting them over explicit channels."""
        dtype = self.feedback.dtype
        device = self.feedback.device
        feedback_tensor = torch.as_tensor(feedback, dtype=dtype, device=device)
        control_tensor = torch.as_tensor(control, dtype=dtype, device=device)
        bias_tensor = torch.as_tensor(bias, dtype=dtype, device=device)

        if self.channels is None:
            expected = (2 * self.radius + 1,)
            if tuple(feedback_tensor.shape) != expected:
                raise ValueError(f"feedback must have shape {expected}.")
            if tuple(control_tensor.shape) != expected:
                raise ValueError(f"control must have shape {expected}.")
            if bias_tensor.numel() != 1:
                raise ValueError("bias must be scalar for a scalar network.")
            loaded_feedback = feedback_tensor
            loaded_control = control_tensor
            loaded_bias = bias_tensor.reshape(())
        else:
            span = 2 * self.radius + 1
            loaded_feedback = self._expand_channel_template(
                feedback_tensor, span, "feedback"
            )
            loaded_control = self._expand_channel_template(
                control_tensor, span, "control"
            )
            if bias_tensor.numel() == 1:
                loaded_bias = bias_tensor.reshape(1).expand(self.channels)
            elif tuple(bias_tensor.shape) == (self.channels,):
                loaded_bias = bias_tensor
            else:
                raise ValueError(
                    f"bias must be scalar or have shape ({self.channels},)."
                )

        with torch.no_grad():
            self.feedback.copy_(loaded_feedback)
            self.control.copy_(loaded_control)
            self.bias.copy_(loaded_bias)

    def _expand_channel_template(
        self, value: torch.Tensor, span: int, name: str
    ) -> torch.Tensor:
        if tuple(value.shape) == (span,):
            return value.reshape(span, 1).expand(span, self.channels)
        expected = (span, self.channels)
        if tuple(value.shape) == expected:
            return value
        raise ValueError(f"{name} must have shape ({span},) or {expected}.")

    def to_template(self, name: str = "learned") -> Template:
        """Detach the learned scalar template for the reference simulator."""
        if self.channels not in {None, 1}:
            raise ValueError(
                "Only scalar or single-channel networks can be converted "
                "to the classical Template contract."
            )
        feedback = self.feedback.detach().cpu().numpy()
        control = self.control.detach().cpu().numpy()
        bias = self.bias.detach().cpu().numpy()
        return Template(
            name=name,
            feedback=feedback.reshape(-1),
            control=control.reshape(-1),
            bias=float(bias.reshape(-1)[0]),
        )

    def _prepare_field(self, value: Any, name: str) -> torch.Tensor:
        field = torch.as_tensor(
            value, dtype=self.feedback.dtype, device=self.feedback.device
        )
        if field.ndim < 1:
            raise ValueError(f"{name} must have at least one dimension.")
        if self.channels is not None:
            if field.ndim < 2 or field.shape[-1] != self.channels:
                raise ValueError(
                    f"{name} must end in a channel axis of size "
                    f"{self.channels}."
                )
        return field

    def drive(self, state: Any, input: Any) -> torch.Tensor:
        """Return the non-decay term ``A*y(x) + B*u + z``."""
        state_tensor = self._prepare_field(state, "state")
        input_tensor = self._prepare_field(input, "input")
        if state_tensor.shape != input_tensor.shape:
            raise ValueError("state and input must have the same shape.")
        return local_drive(
            state=state_tensor,
            input_array=input_tensor,
            feedback=self.feedback,
            control=self.control,
            bias=self.bias,
            activation=self._activation_fn,
            backend=self.backend,
            boundary=self.boundary,
            boundary_value=self.boundary_value,
        )

    def derivative(self, state: Any, input: Any) -> torch.Tensor:
        """Return ``dx/dt`` using the library's canonical ODE definition."""
        state_tensor = self._prepare_field(state, "state")
        input_tensor = self._prepare_field(input, "input")
        if state_tensor.shape != input_tensor.shape:
            raise ValueError("state and input must have the same shape.")
        return compute_derivative(
            state=state_tensor,
            input_array=input_tensor,
            feedback=self.feedback,
            control=self.control,
            bias=self.bias,
            activation=self._activation_fn,
            backend=self.backend,
            boundary=self.boundary,
            boundary_value=self.boundary_value,
        )

    def step(
        self,
        state: Any,
        input: Any,
        extra_drive: Any | None = None,
    ) -> torch.Tensor:
        """Advance one step, optionally adding caller-owned channel mixing."""
        state_tensor = self._prepare_field(state, "state")
        input_tensor = self._prepare_field(input, "input")
        if state_tensor.shape != input_tensor.shape:
            raise ValueError("state and input must have the same shape.")
        if self.method != "euler" and extra_drive is not None:
            raise ValueError(
                "extra_drive is currently supported only by Euler."
            )

        term = (
            self.derivative(state_tensor, input_tensor)
            if self.method == "euler"
            else self.drive(state_tensor, input_tensor)
        )
        if extra_drive is not None:
            extra = self._prepare_field(extra_drive, "extra_drive")
            if extra.shape != state_tensor.shape:
                raise ValueError(
                    "extra_drive and state must have the same shape."
                )
            term = term + extra
        return STEPPERS[self.method](state_tensor, self.dt, term)

    def forward(self, input: Any, state: Any | None = None) -> torch.Tensor:
        """Evolve ``state`` for the configured number of integration steps."""
        input_tensor = self._prepare_field(input, "input")
        current = (
            torch.zeros_like(input_tensor)
            if state is None
            else self._prepare_field(state, "state")
        )
        if current.shape != input_tensor.shape:
            raise ValueError("state and input must have the same shape.")

        for _ in range(self.steps):
            current = self.step(current, input_tensor)
        return current


__all__ = ["DifferentiableCellularNetwork"]
