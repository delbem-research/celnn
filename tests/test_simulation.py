import numpy as np
import pytest

from celnn import CellularNetwork, SimulationConfig
from celnn.core.exceptions import SolverError


def test_simulation_trajectory_shapes():
    signal = np.ones(6)
    net = CellularNetwork(
        input=signal,
        feedback=[0.0, 0.0, 0.0],
        control=[0.0, 1.0, 0.0],
        activation="identity",
    )
    result = net.run(
        SimulationConfig(t_end=0.3, dt=0.1, return_trajectory=True)
    )
    assert result.trajectory_state is not None
    assert result.trajectory_output is not None
    assert result.trajectory_state.shape == (4, 6)
    assert result.trajectory_output.shape == (4, 6)
    assert result.time.shape == (4,)


def test_store_every_reduces_trajectory_length():
    signal = np.ones(6)
    net = CellularNetwork(
        input=signal,
        feedback=[0.0, 0.0, 0.0],
        control=[0.0, 1.0, 0.0],
        activation="identity",
    )
    result = net.run(
        SimulationConfig(
            t_end=0.4, dt=0.1, return_trajectory=True, store_every=2
        )
    )
    assert result.trajectory_state is not None
    assert result.trajectory_state.shape[0] == 3


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("dt", np.nan),
        ("dt", np.inf),
        ("t_start", np.nan),
        ("t_end", np.inf),
    ],
)
def test_simulation_rejects_non_finite_time_values(field, value):
    with pytest.raises(SolverError, match="finite"):
        SimulationConfig(**{field: value})


@pytest.mark.parametrize("store_every", [0, -1, 1.5, True])
def test_store_every_requires_a_positive_integer(store_every):
    with pytest.raises(SolverError, match="positive integer"):
        SimulationConfig(store_every=store_every)
