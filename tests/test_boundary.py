import numpy as np
import pytest

from celnn import CellularNetwork, SimulationConfig
from celnn.backends.cupy_backend import CuPyBackend
from celnn.backends.numpy_backend import NumPyBackend
from celnn.core.boundary import normalize_boundary_mode
from celnn.core.exceptions import CelNNError


def test_boundary_mode_validation():
    assert normalize_boundary_mode("Reflect") == "reflect"
    with pytest.raises(CelNNError):
        normalize_boundary_mode("invalid")


def test_numpy_backend_boundary_modes():
    backend = NumPyBackend()
    array = np.array([1.0, 2.0, 3.0])
    kernel = np.array([1.0, 0.0, 0.0])

    constant = backend.aggregate_local(
        array, kernel, mode="constant", cval=0.0
    )
    wrap = backend.aggregate_local(array, kernel, mode="wrap", cval=0.0)
    reflect = backend.aggregate_local(array, kernel, mode="reflect", cval=0.0)
    nearest = backend.aggregate_local(array, kernel, mode="nearest", cval=0.0)

    assert np.allclose(constant, [0.0, 1.0, 2.0])
    assert np.allclose(wrap, [3.0, 1.0, 2.0])
    assert np.allclose(reflect, [2.0, 1.0, 2.0])
    assert np.allclose(nearest, [1.0, 1.0, 2.0])


def test_cupy_backend_matches_numpy_when_available():
    if not CuPyBackend.is_available():
        pytest.skip("CuPy/CUDA is not available.")

    numpy_backend = NumPyBackend()
    cupy_backend = CuPyBackend()
    array = np.arange(9.0).reshape(3, 3)
    kernel = np.array(
        [
            [0.0, 1.0, 0.0],
            [1.0, 2.0, 1.0],
            [0.0, 1.0, 0.0],
        ]
    )

    expected = numpy_backend.aggregate_local(
        array,
        kernel,
        mode="reflect",
        cval=0.0,
    )
    actual = cupy_backend.aggregate_local(
        array,
        kernel,
        mode="reflect",
        cval=0.0,
    )
    assert np.allclose(actual, expected)


def test_network_runs_on_gpu_when_available():
    if not CuPyBackend.is_available():
        pytest.skip("CuPy/CUDA is not available.")

    signal = np.ones(8)
    net = CellularNetwork(
        input=signal,
        feedback=[0.0, 0.0, 0.0],
        control=[0.0, 1.0, 0.0],
        activation="identity",
        boundary="reflect",
        device="gpu",
    )
    result = net.run(SimulationConfig(t_end=0.2, dt=0.1))
    assert result.metadata["backend"] == "cupy"
    assert result.state.shape == signal.shape
