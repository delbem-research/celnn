"""Simulation configuration dataclass."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any

import numpy as np

from .exceptions import SolverError


@dataclass(slots=True)
class SimulationConfig:
    """Configuration for a network simulation."""

    t_start: float = 0.0
    t_end: float = 1.0
    dt: float = 0.01
    solver: str = "euler"
    return_trajectory: bool = False
    store_every: int = 1
    stability_checks: bool = True
    progress: bool = False

    def __post_init__(self) -> None:
        if not isfinite(self.dt) or self.dt <= 0:
            raise SolverError(
                f"dt must be finite and positive, got {self.dt}."
            )
        if not isfinite(self.t_start) or not isfinite(self.t_end):
            raise SolverError(
                "t_start and t_end must be finite, "
                f"got {self.t_start} -> {self.t_end}."
            )
        if self.t_end < self.t_start:
            raise SolverError(
                "t_end must be greater than or equal to t_start, "
                f"got {self.t_start} -> {self.t_end}."
            )
        if (
            not isinstance(self.store_every, int)
            or isinstance(self.store_every, bool)
            or self.store_every <= 0
        ):
            raise SolverError(
                "store_every must be a positive integer, "
                f"got {self.store_every}."
            )
        self.solver = self.solver.lower().strip()

    def time_points(self) -> np.ndarray:
        """Return solver step points including the final time."""
        span = self.t_end - self.t_start
        if span == 0:
            return np.array([self.t_start], dtype=float)
        steps = int(np.floor(span / self.dt))
        times = self.t_start + self.dt * np.arange(steps + 1, dtype=float)
        if times[-1] < self.t_end:
            times = np.concatenate(
                [times, np.array([self.t_end], dtype=float)]
            )
        else:
            times[-1] = self.t_end
        return times

    def to_dict(self) -> dict[str, Any]:
        """Serialize the configuration to a JSON-friendly dictionary."""
        return {
            "t_start": self.t_start,
            "t_end": self.t_end,
            "dt": self.dt,
            "solver": self.solver,
            "return_trajectory": self.return_trajectory,
            "store_every": self.store_every,
            "stability_checks": self.stability_checks,
            "progress": self.progress,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SimulationConfig":
        """Restore a configuration from a dictionary."""
        return cls(
            t_start=float(data.get("t_start", 0.0)),
            t_end=float(data.get("t_end", 1.0)),
            dt=float(data.get("dt", 0.01)),
            solver=data.get("solver", "euler"),
            return_trajectory=bool(data.get("return_trajectory", False)),
            store_every=int(data.get("store_every", 1)),
            stability_checks=bool(data.get("stability_checks", True)),
            progress=bool(data.get("progress", False)),
        )
