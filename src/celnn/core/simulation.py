"""Simulation configuration dataclass."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .exceptions import SolverError

_ALLOWED_KEYS = {
    "t_start",
    "t_end",
    "dt",
    "solver",
    "return_trajectory",
    "store_every",
    "progress",
}


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SolverError(f"{name} must be a real number.")
    return float(value)


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise SolverError(f"{name} must be a boolean.")
    return value


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SolverError(f"{name} must be an integer.")
    return value


@dataclass(slots=True)
class SimulationConfig:
    """Configuration for a network simulation."""

    t_start: float = 0.0
    t_end: float = 1.0
    dt: float = 0.01
    solver: str = "euler"
    return_trajectory: bool = False
    store_every: int = 1
    progress: bool = False

    def __post_init__(self) -> None:
        if self.dt <= 0:
            raise SolverError(f"dt must be positive, got {self.dt}.")
        if self.t_end < self.t_start:
            raise SolverError(
                "t_end must be greater than or equal to t_start, "
                f"got {self.t_start} -> {self.t_end}."
            )
        if self.store_every <= 0:
            raise SolverError(
                "store_every must be a positive integer, "
                f"got {self.store_every}."
            )
        if not isinstance(self.solver, str):
            raise SolverError("solver must be a string.")
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
            "progress": self.progress,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SimulationConfig":
        """Restore a validated configuration from a dictionary."""
        if not isinstance(data, dict):
            raise SolverError("SimulationConfig data must be a dictionary.")
        unknown = set(data) - _ALLOWED_KEYS
        if unknown:
            names = ", ".join(sorted(unknown))
            raise SolverError(f"Unknown SimulationConfig fields: {names}.")

        solver = data.get("solver", "euler")
        if not isinstance(solver, str):
            raise SolverError("solver must be a string.")
        return cls(
            t_start=_number(data.get("t_start", 0.0), "t_start"),
            t_end=_number(data.get("t_end", 1.0), "t_end"),
            dt=_number(data.get("dt", 0.01), "dt"),
            solver=solver,
            return_trajectory=_boolean(
                data.get("return_trajectory", False), "return_trajectory"
            ),
            store_every=_integer(data.get("store_every", 1), "store_every"),
            progress=_boolean(data.get("progress", False), "progress"),
        )
