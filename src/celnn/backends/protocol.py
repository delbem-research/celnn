"""Backend protocol for array operations."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np


class ArrayBackend(Protocol):
    """Protocol for backend implementations."""

    name: str

    def aggregate_local(
        self,
        values: np.ndarray,
        weights: np.ndarray,
        *,
        mode: str,
        cval: float = 0.0,
    ) -> np.ndarray | Any:
        """Aggregate local neighborhoods using a stencil."""
