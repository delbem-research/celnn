import numpy as np
import pytest

from celnn import CellularNetwork, SimulationConfig
from celnn.core.exceptions import CelNNError, ShapeMismatchError
from celnn.core.steppers import euler_step
from celnn.templates import Template


def test_network_creation_defaults():
    signal = np.ones(8)
    net = CellularNetwork(input=signal)
    assert net.state.shape == signal.shape
    assert net.feedback.shape == (3,)
    assert net.control.shape == (3,)
    assert np.allclose(net.state, 0.0)
    assert net.backend.name == "numpy"
    assert net.dtype == np.dtype(np.float64)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_network_preserves_supported_dtype(dtype):
    signal = np.ones(8, dtype=dtype)
    net = CellularNetwork(
        input=signal,
        feedback=np.array([0.0, 0.0, 0.0], dtype=dtype),
        control=np.array([0.0, 1.0, 0.0], dtype=dtype),
        bias=np.array(0.25, dtype=dtype),
        activation="identity",
        dtype=dtype,
    )
    result = net.run(SimulationConfig(t_end=0.2, dt=0.1))
    assert net.input.dtype == np.dtype(dtype)
    assert net.feedback.dtype == np.dtype(dtype)
    assert net.control.dtype == np.dtype(dtype)
    assert net.bias.dtype == np.dtype(dtype)
    assert result.state.dtype == np.dtype(dtype)
    assert result.output.dtype == np.dtype(dtype)


@pytest.mark.parametrize(
    "dtype", [np.float16, np.int64, np.bool_, np.complex128]
)
def test_network_preserves_legacy_numpy_dtype_acceptance(dtype):
    net = CellularNetwork(input=np.ones(4), dtype=dtype)
    assert net.dtype == np.dtype(dtype)
    assert net.input.dtype == np.dtype(dtype)


def test_network_rejects_mismatched_initial_state_shape():
    signal = np.ones(8)
    with pytest.raises(ShapeMismatchError):
        CellularNetwork(input=signal, initial_state=np.zeros(4))


def test_from_template_uses_template_fields():
    template = Template(
        name="demo",
        feedback=[0.0, 1.0, 0.0],
        control=[0.0, 1.0, 0.0],
        bias=0.25,
    )
    signal = np.ones(5)
    net = CellularNetwork.from_template(template=template, input=signal)
    assert np.allclose(net.feedback, np.array([0.0, 1.0, 0.0]))
    assert np.allclose(net.control, np.array([0.0, 1.0, 0.0]))
    assert np.allclose(net.bias, 0.25)


def test_step_uses_canonical_euler_semantics():
    signal = np.ones(5)
    net = CellularNetwork(
        input=signal,
        feedback=[0.0, 0.0, 0.0],
        control=[0.0, 1.0, 0.0],
        activation="identity",
    )
    before = net.state.copy()
    derivative = net.derivative(before)
    expected = euler_step(before, 0.1, derivative)
    actual = net.step(0.1)
    assert np.allclose(actual, expected)


@pytest.mark.parametrize("dt", [np.nan, np.inf, -np.inf])
def test_step_rejects_non_finite_dt_without_mutating_state(dt):
    net = CellularNetwork([1.0])
    before = net.state.copy()

    with pytest.raises(CelNNError, match="finite and positive"):
        net.step(dt)

    assert np.array_equal(net.state, before)


def test_reset_restores_initial_state():
    signal = np.ones(5)
    net = CellularNetwork(
        input=signal,
        initial_state=np.linspace(-1.0, 1.0, 5),
        feedback=[0.0, 0.0, 0.0],
        control=[0.0, 1.0, 0.0],
    )
    net.step(0.1)
    net.reset()
    assert np.allclose(net.state, np.linspace(-1.0, 1.0, 5))


def test_run_returns_result_with_convergence_contract():
    signal = np.ones(5)
    net = CellularNetwork(
        input=signal,
        feedback=[0.0, 0.0, 0.0],
        control=[0.0, 1.0, 0.0],
        activation="identity",
    )
    result = net.run(SimulationConfig(t_end=0.2, dt=0.1))
    assert result.state.shape == signal.shape
    assert result.output.shape == signal.shape
    assert result.metadata["backend"] == "numpy"
    assert result.convergence is not None
    assert set(result.convergence) == {
        "max_abs_state_delta",
        "approx_converged",
    }


def test_stability_checks_preserve_warning_contract():
    signal = np.ones(5)
    net = CellularNetwork(
        input=signal,
        feedback=[0.0, 0.0, 0.0],
        control=[0.0, 1.0, 0.0],
        activation="identity",
        metadata={"source": "test"},
    )
    result = net.run(SimulationConfig(t_end=2.0, dt=1.1))
    assert result.metadata["warnings"] == [
        "dt > 1.0 may be numerically unstable for explicit schemes."
    ]
    assert net.metadata == {"source": "test"}


def test_network_accepts_auto_device():
    signal = np.ones(5)
    net = CellularNetwork(input=signal, device="auto")
    result = net.run(SimulationConfig(t_end=0.1, dt=0.1))
    assert result.state.shape == signal.shape
    assert result.metadata["backend"] in {"numpy", "cupy"}
