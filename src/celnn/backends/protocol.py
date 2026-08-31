"""Backend protocol for array operations."""

from __future__ import annotations

from typing import Any, Protocol


class ArrayBackend(Protocol):
    """Protocol for backend implementations."""

    name: str

    def aggregate_local(
        self,
        values: Any,
        weights: Any,
        *,
        mode: str,
        cval: float = 0.0,
    ) -> Any:
        """Aggregate local neighborhoods using a stencil."""
