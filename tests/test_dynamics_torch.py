"""The reference dynamics module must run unmodified on tensors.

These tests are the point of the whole exercise: ``celnn.core.dynamics`` is
written once, and the differentiable path reuses it rather than restating it.
If these pass, there is exactly one definition of the CelNN ODE in the library.
"""

from __future__ import annotations

import numpy as np
import pytest

from celnn.backends.numpy_backend import NumPyBackend
from celnn.core.activations import resolve_activation
from celnn.core.boundary import VALID_BOUNDARY_MODES
from celnn.core.dynamics import derivative, local_drive

torch = pytest.importorskip("torch", reason="PyTorch is optional")

from celnn.backends.torch_backend import TorchBackend  # noqa: E402

FEEDBACK = np.array([0.1, 0.5, 1.0, 0.5, 0.1])
CONTROL = np.array([0.0, 0.2, 0.7, 0.2, 0.0])
BIAS = 0.3


def _numpy_derivative(state, u, activation, mode):
    return derivative(
        state=state,
        input_array=u,
        feedback=FEEDBACK,
        control=CONTROL,
        bias=np.asarray(BIAS),
        activation=resolve_activation(activation),
        backend=NumPyBackend(),
        boundary=mode,
        boundary_value=0.0,
    )


def _torch_derivative(state, u, activation, mode):
    as_t = lambda a: torch.tensor(a, dtype=torch.float64)  # noqa: E731
    return derivative(
        state=as_t(state),
        input_array=as_t(u),
        feedback=as_t(FEEDBACK),
        control=as_t(CONTROL),
        bias=torch.tensor(BIAS, dtype=torch.float64),
        activation=resolve_activation(activation),
        backend=TorchBackend(),
        boundary=mode,
        boundary_value=0.0,
    )


@pytest.mark.parametrize("mode", VALID_BOUNDARY_MODES)
@pytest.mark.parametrize(
    "activation", ["piecewise_linear", "tanh_activation", "identity"]
)
def test_reference_derivative_agrees_on_tensors(mode, activation):
    rng = np.random.default_rng(0)
    state = rng.normal(size=21)
    u = rng.normal(size=21)

    expected = _numpy_derivative(state, u, activation, mode)
    got = _torch_derivative(state, u, activation, mode)

    np.testing.assert_allclose(
        got.numpy(), expected, rtol=1e-12, atol=1e-12
    )


def test_local_drive_also_reuses_cleanly():
    rng = np.random.default_rng(1)
    state = rng.normal(size=12)
    u = rng.normal(size=12)

    expected = local_drive(
        state=state,
        input_array=u,
        feedback=FEEDBACK,
        control=CONTROL,
        bias=np.asarray(BIAS),
        activation=resolve_activation("piecewise_linear"),
        backend=NumPyBackend(),
        boundary="constant",
        boundary_value=0.0,
    )
    got = local_drive(
        state=torch.tensor(state, dtype=torch.float64),
        input_array=torch.tensor(u, dtype=torch.float64),
        feedback=torch.tensor(FEEDBACK, dtype=torch.float64),
        control=torch.tensor(CONTROL, dtype=torch.float64),
        bias=torch.tensor(BIAS, dtype=torch.float64),
        activation=resolve_activation("piecewise_linear"),
        backend=TorchBackend(),
        boundary="constant",
        boundary_value=0.0,
    )

    np.testing.assert_allclose(got.numpy(), expected, rtol=1e-12, atol=1e-12)


def test_leak_term_dominates_when_templates_vanish():
    state = torch.randn(9, dtype=torch.float64)
    zeros = torch.zeros(3, dtype=torch.float64)
    got = derivative(
        state=state,
        input_array=torch.zeros_like(state),
        feedback=zeros,
        control=zeros,
        bias=torch.zeros((), dtype=torch.float64),
        activation=resolve_activation("identity"),
        backend=TorchBackend(),
        boundary="constant",
        boundary_value=0.0,
    )
    torch.testing.assert_close(got, -state)


def test_gradients_reach_the_templates_through_the_reference_module():
    feedback = torch.tensor(
        FEEDBACK, dtype=torch.float64, requires_grad=True
    )
    control = torch.tensor(CONTROL, dtype=torch.float64, requires_grad=True)
    bias = torch.tensor(BIAS, dtype=torch.float64, requires_grad=True)
    state = torch.randn(16, dtype=torch.float64, requires_grad=True)

    result = derivative(
        state=state,
        input_array=torch.randn(16, dtype=torch.float64),
        feedback=feedback,
        control=control,
        bias=bias,
        activation=resolve_activation("tanh_activation"),
        backend=TorchBackend(),
        boundary="wrap",
        boundary_value=0.0,
    )
    result.pow(2).sum().backward()

    for name, tensor in (
        ("feedback", feedback),
        ("control", control),
        ("bias", bias),
        ("state", state),
    ):
        assert tensor.grad is not None, name
        assert torch.isfinite(tensor.grad).all(), name
        assert tensor.grad.abs().sum() > 0, name


def test_multichannel_derivative_keeps_channels_independent():
    """Diagonal templates: channel c never reads channel c'."""
    backend = TorchBackend()
    field = torch.zeros(1, 10, 4, dtype=torch.float64)
    feedback = torch.randn(3, 4, dtype=torch.float64)
    control = torch.zeros(3, 4, dtype=torch.float64)

    def run(state):
        return derivative(
            state=state,
            input_array=torch.zeros_like(state),
            feedback=feedback,
            control=control,
            bias=torch.zeros(4, dtype=torch.float64),
            activation=resolve_activation("identity"),
            backend=backend,
            boundary="constant",
            boundary_value=0.0,
        )

    base = run(field)
    perturbed = field.clone()
    perturbed[0, 5, 0] = 1.0
    changed = (run(perturbed) - base).abs().sum(dim=(0, 1))

    assert changed[0] > 0
    torch.testing.assert_close(
        changed[1:], torch.zeros(3, dtype=torch.float64)
    )
