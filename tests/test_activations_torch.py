"""Activations must be one implementation serving every array library."""

from __future__ import annotations

import numpy as np
import pytest

from celnn.core.activations import ACTIVATIONS, resolve_activation

torch = pytest.importorskip("torch", reason="PyTorch is optional")

SAMPLE = [-3.0, -1.5, -1.0, -0.25, 0.0, 0.25, 1.0, 1.5, 3.0]
DIFFERENTIABLE = (
    "piecewise_linear",
    "saturated_linear",
    "identity",
    "tanh_activation",
    "sigmoid_activation",
    "relu_activation",
)


@pytest.mark.parametrize("name", sorted(ACTIVATIONS))
def test_torch_result_matches_the_numpy_result(name):
    activation = resolve_activation(name)
    expected = activation(np.array(SAMPLE, dtype=float))
    got = activation(torch.tensor(SAMPLE, dtype=torch.float64))

    assert isinstance(got, torch.Tensor), f"{name} left the torch namespace"
    np.testing.assert_allclose(
        got.detach().numpy(), expected, rtol=1e-12, atol=1e-12
    )


@pytest.mark.parametrize("name", sorted(ACTIVATIONS))
def test_torch_dtype_and_device_are_preserved(name):
    activation = resolve_activation(name)
    got = activation(torch.tensor(SAMPLE, dtype=torch.float32))
    assert got.dtype == torch.float32


@pytest.mark.parametrize("name", DIFFERENTIABLE)
def test_gradients_survive_the_activation(name):
    activation = resolve_activation(name)
    x = torch.tensor([0.3, -0.7, 1.4], dtype=torch.float64, requires_grad=True)

    activation(x).sum().backward()

    assert x.grad is not None, f"{name} produced no gradient"
    assert torch.isfinite(x.grad).all(), f"{name} produced a non-finite grad"


def test_numpy_behaviour_is_unchanged_for_list_input():
    """Lists are still coerced, as every existing caller expects."""
    got = resolve_activation("piecewise_linear")([-2, 0, 2])
    assert isinstance(got, np.ndarray)
    np.testing.assert_allclose(got, [-1.0, 0.0, 1.0])


def test_piecewise_linear_saturates_in_torch():
    activation = resolve_activation("piecewise_linear")
    got = activation(torch.tensor([-9.0, 0.5, 9.0], dtype=torch.float64))
    np.testing.assert_allclose(got.numpy(), [-1.0, 0.5, 1.0])
