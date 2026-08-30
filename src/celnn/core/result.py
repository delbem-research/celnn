"""Simulation result dataclass."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(slots=True)
class SimulationResult:
    """Container returned by ``CellularNetwork.run``."""

    state: np.ndarray
    output: np.ndarray
    time: np.ndarray
    trajectory_state: np.ndarray | None = None
    trajectory_output: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    convergence: dict[str, Any] | None = None

    @property
    def has_trajectory(self) -> bool:
        """Return whether trajectory arrays were stored."""
        return (
            self.trajectory_state is not None
            and self.trajectory_output is not None
        )

    @property
    def final_time(self) -> float:
        """Return the final stored time."""
        return float(np.asarray(self.time)[-1])
