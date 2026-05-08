import numpy as np
import pytest

from celnn import CellularNetwork, SimulationConfig
from celnn.core.exceptions import ShapeMismatchError
from celnn.templates import Template


def test_network_creation_defaults():
    signal = np.ones(8)
    net = CellularNetwork(input=signal)
    assert net.state.shape == signal.shape
    assert net.feedback.shape == (3,)
    assert net.control.shape == (3,)
    assert np.allclose(net.state, 0.0)
    assert net.backend.name == "numpy"


def test_network_rejects_mismatched_state_shape():
    signal = np.ones(8)
    with pytest.raises(ShapeMismatchError):
        CellularNetwork(input=signal, state_shape=(4,))


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


def test_run_returns_result():
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


def test_network_accepts_auto_device():
    signal = np.ones(5)
    net = CellularNetwork(input=signal, device="auto")
    result = net.run(SimulationConfig(t_end=0.1, dt=0.1))
    assert result.state.shape == signal.shape
    assert result.metadata["backend"] in {"numpy", "cupy"}
