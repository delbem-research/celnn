import numpy as np
import pytest

from celnn import CellularNetwork, SimulationConfig
from celnn.core.exceptions import SolverError


def make_decay_network(dtype=np.float64):
    signal = np.ones(5, dtype=dtype)
    return CellularNetwork(
        input=signal,
        feedback=[0.0, 0.0, 0.0],
        control=[0.0, 1.0, 0.0],
        activation="identity",
        boundary="reflect",
        dtype=dtype,
    )


def test_euler_solver_matches_expected_decay():
    net = make_decay_network()
    result = net.run(SimulationConfig(t_end=0.3, dt=0.1, solver="euler"))
    assert np.allclose(result.state, np.full(5, 0.271), atol=1e-6)


def test_semi_implicit_solver_runs():
    net = make_decay_network()
    result = net.run(
        SimulationConfig(t_end=0.3, dt=0.1, solver="semi_implicit_euler")
    )
    assert np.allclose(result.state, np.full(5, 0.2486852), atol=1e-6)


def test_scipy_solver_if_available():
    pytest.importorskip("scipy")
    net = make_decay_network()
    result = net.run(
        SimulationConfig(t_end=0.3, dt=0.1, solver="solve_ivp")
    )
    assert np.allclose(
        result.state, np.full(5, 1.0 - np.exp(-0.3)), atol=1e-3
    )


def test_scipy_solver_rejects_float32_before_execution():
    net = make_decay_network(np.float32)
    with pytest.raises(SolverError, match="requires a float64"):
        net.run(SimulationConfig(t_end=0.3, dt=0.1, solver="solve_ivp"))
