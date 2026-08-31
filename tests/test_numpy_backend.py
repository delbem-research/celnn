import numpy as np
import pytest

from celnn.backends.numpy_backend import NumPyBackend


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_numpy_backend_preserves_supported_float_dtype(dtype):
    backend = NumPyBackend()
    values = np.arange(7, dtype=dtype)
    weights = np.array([0.25, 0.5, 0.25], dtype=dtype)

    result = backend.aggregate_local(
        values,
        weights,
        mode="reflect",
    )

    assert result.dtype == np.dtype(dtype)
