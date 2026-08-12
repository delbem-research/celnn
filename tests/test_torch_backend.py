"""The torch backend must aggregate exactly as the reference one does."""

from __future__ import annotations

import numpy as np
import pytest

from celnn.backends.numpy_backend import NumPyBackend
from celnn.core.boundary import VALID_BOUNDARY_MODES

torch = pytest.importorskip("torch", reason="PyTorch is optional")

from celnn.backends.torch_backend import TorchBackend  # noqa: E402

KERNEL = np.array([0.2, -0.5, 1.0, -0.5, 0.2])


@pytest.mark.parametrize("mode", VALID_BOUNDARY_MODES)
def test_matches_the_numpy_backend_for_every_boundary_mode(mode):
    rng = np.random.default_rng(0)
    signal = rng.normal(size=19)

    expected = NumPyBackend().aggregate_local(
        signal, KERNEL, mode=mode, cval=0.0
    )
    got = TorchBackend().aggregate_local(
        torch.tensor(signal, dtype=torch.float64),
        torch.tensor(KERNEL, dtype=torch.float64),
        mode=mode,
        cval=0.0,
    )

    np.testing.assert_allclose(got.numpy(), expected, rtol=1e-12, atol=1e-12)


def test_constant_mode_honours_a_nonzero_fill():
    expected = NumPyBackend().aggregate_local(
        np.ones(5), KERNEL, mode="constant", cval=3.0
    )
    got = TorchBackend().aggregate_local(
        torch.ones(5, dtype=torch.float64),
        torch.tensor(KERNEL, dtype=torch.float64),
        mode="constant",
        cval=3.0,
    )
    np.testing.assert_allclose(got.numpy(), expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("mode", VALID_BOUNDARY_MODES)
def test_batched_input_matches_the_reference_row_by_row(mode):
    rng = np.random.default_rng(1)
    batch = rng.normal(size=(4, 13))

    expected = np.stack(
        [
            NumPyBackend().aggregate_local(row, KERNEL, mode=mode, cval=0.0)
            for row in batch
        ]
    )
    got = TorchBackend().aggregate_local(
        torch.tensor(batch, dtype=torch.float64),
        torch.tensor(KERNEL, dtype=torch.float64),
        mode=mode,
        cval=0.0,
    )

    np.testing.assert_allclose(got.numpy(), expected, rtol=1e-12, atol=1e-12)


def test_diagonal_template_applies_one_weight_vector_per_channel():
    """A (span, channels) kernel is a diagonal template: no channel mixing."""
    rng = np.random.default_rng(2)
    field = rng.normal(size=(2, 11, 3))
    kernel = rng.normal(size=(5, 3))

    expected = np.stack(
        [
            np.stack(
                [
                    NumPyBackend().aggregate_local(
                        sample[:, channel],
                        kernel[:, channel],
                        mode="constant",
                        cval=0.0,
                    )
                    for channel in range(3)
                ],
                axis=-1,
            )
            for sample in field
        ]
    )

    got = TorchBackend().aggregate_local(
        torch.tensor(field, dtype=torch.float64),
        torch.tensor(kernel, dtype=torch.float64),
        mode="constant",
        cval=0.0,
    )

    np.testing.assert_allclose(got.numpy(), expected, rtol=1e-12, atol=1e-12)


def test_single_channel_reduces_to_the_scalar_case():
    """d = 1 must recover the classical CelNN aggregation exactly."""
    rng = np.random.default_rng(3)
    signal = rng.normal(size=15)

    scalar = TorchBackend().aggregate_local(
        torch.tensor(signal, dtype=torch.float64),
        torch.tensor(KERNEL, dtype=torch.float64),
        mode="reflect",
    )
    channelled = TorchBackend().aggregate_local(
        torch.tensor(signal, dtype=torch.float64).reshape(1, 15, 1),
        torch.tensor(KERNEL, dtype=torch.float64).reshape(5, 1),
        mode="reflect",
    )

    np.testing.assert_allclose(
        channelled.reshape(-1).numpy(), scalar.numpy(), rtol=1e-12, atol=1e-12
    )


def test_gradients_flow_to_both_field_and_template():
    field = torch.randn(2, 9, 3, dtype=torch.float64, requires_grad=True)
    kernel = torch.randn(3, 3, dtype=torch.float64, requires_grad=True)

    result = TorchBackend().aggregate_local(field, kernel, mode="constant")
    result.sum().backward()

    assert field.grad is not None and field.grad.abs().sum() > 0
    assert kernel.grad is not None and kernel.grad.abs().sum() > 0


def test_device_and_dtype_are_preserved():
    got = TorchBackend().aggregate_local(
        torch.ones(7, dtype=torch.float32),
        torch.ones(3, dtype=torch.float32),
        mode="constant",
    )
    assert got.dtype == torch.float32
