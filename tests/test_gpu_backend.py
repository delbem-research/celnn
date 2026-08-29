import numpy as np
import pytest

from celnn import CellularNetwork, SimulationConfig
from celnn.backends import get_backend
from celnn.backends.cupy_backend import CuPyBackend
from celnn.backends.numpy_backend import NumPyBackend
from celnn.core.exceptions import BackendError


class _FakeCuPy:
    float32 = np.float32
    float64 = np.float64

    @staticmethod
    def asarray(values, dtype=None):
        return np.asarray(values, dtype=dtype)

    @staticmethod
    def pad(array, pad_width, mode, constant_values=0.0):
        kwargs = {"mode": mode}
        if mode == "constant":
            kwargs["constant_values"] = constant_values
        return np.pad(array, pad_width, **kwargs)

    @staticmethod
    def zeros_like(array, dtype=None):
        return np.zeros_like(array, dtype=dtype)

    @staticmethod
    def asnumpy(array):
        return np.asarray(array)

    @staticmethod
    def flip(array):
        return np.flip(array)

    @staticmethod
    def arange(*args, **kwargs):
        return np.arange(*args, **kwargs)

    @staticmethod
    def sum(array):
        return np.asarray(array).sum()


@pytest.fixture()
def stubbed_cupy_backend(monkeypatch):
    monkeypatch.setattr(CuPyBackend, "_import_cupy", lambda self: _FakeCuPy())
    return CuPyBackend()


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_cupy_backend_1d_matches_numpy_and_preserves_dtype(
    stubbed_cupy_backend, dtype
):
    numpy_backend = NumPyBackend()
    array = np.array([1.0, 2.0, 3.0, 4.0], dtype=dtype)
    kernel = np.array([0.25, 0.5, 0.25], dtype=dtype)
    for mode in ("constant", "wrap", "reflect", "nearest", "mirror"):
        expected = numpy_backend.aggregate_local(array, kernel, mode=mode)
        actual = stubbed_cupy_backend.aggregate_local(array, kernel, mode=mode)
        assert np.allclose(actual, expected)
        assert actual.dtype == np.dtype(dtype)


def test_cupy_backend_2d_matches_numpy_with_stub(stubbed_cupy_backend):
    numpy_backend = NumPyBackend()
    array = np.arange(25.0, dtype=np.float32).reshape(5, 5)
    kernel = np.array(
        [
            [0.0, 1.0, 0.0],
            [1.0, 4.0, 1.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    expected = numpy_backend.aggregate_local(array, kernel, mode="reflect")
    actual = stubbed_cupy_backend.aggregate_local(
        array,
        kernel,
        mode="reflect",
    )
    assert np.allclose(actual, expected)
    assert actual.dtype == np.float32


def test_get_backend_auto_falls_back_to_numpy(monkeypatch):
    monkeypatch.setattr(
        CuPyBackend,
        "is_available",
        classmethod(lambda cls: False),
    )
    backend = get_backend("auto")
    assert backend.name == "numpy"


def test_get_backend_gpu_raises_when_cupy_cannot_initialize(monkeypatch):
    def _raise_backend_error(_self):
        raise BackendError("simulated gpu setup failure")

    monkeypatch.setattr(CuPyBackend, "_import_cupy", _raise_backend_error)
    with pytest.raises(BackendError, match="simulated gpu setup failure"):
        get_backend("gpu")


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_network_runs_with_stubbed_gpu_backend(
    stubbed_cupy_backend, dtype
):
    signal = np.ones(8, dtype=dtype)
    net = CellularNetwork(
        input=signal,
        feedback=[0.0, 0.0, 0.0],
        control=[0.0, 1.0, 0.0],
        activation="identity",
        boundary="reflect",
        device="gpu",
        dtype=dtype,
    )
    result = net.run(SimulationConfig(t_end=0.2, dt=0.1))
    assert result.metadata["backend"] == "cupy"
    assert isinstance(result.state, np.ndarray)
    assert isinstance(result.output, np.ndarray)
    assert result.state.dtype == np.dtype(dtype)
    assert result.output.dtype == np.dtype(dtype)
