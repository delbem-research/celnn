"""Optional plotting helpers built on Matplotlib."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable

import numpy as np

from ..core.exceptions import OptionalDependencyError
from ..utils.doc import optional_dependency_message

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


def _plt() -> Any:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - optional dependency branch
        raise OptionalDependencyError(
            optional_dependency_message("matplotlib", "viz", "Visualization")
        ) from exc
    return plt


def plot_signal(
    signal: np.ndarray, *, title: str = "Signal", path: str | None = None
) -> tuple[Figure, Axes]:
    """Plot a one-dimensional signal."""
    plt = _plt()
    fig, ax = plt.subplots()
    ax.plot(np.asarray(signal, dtype=float))
    ax.set_title(title)
    ax.set_xlabel("Index")
    ax.set_ylabel("Value")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
    return fig, ax


def plot_grid(
    grid: np.ndarray, *, title: str = "Grid", path: str | None = None
) -> tuple[Figure, Axes]:
    """Plot a two-dimensional grid."""
    plt = _plt()
    fig, ax = plt.subplots()
    image = ax.imshow(
        np.asarray(grid, dtype=float), cmap="gray", interpolation="nearest"
    )
    ax.set_title(title)
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
    return fig, ax


def plot_trajectory(
    trajectory: Iterable[np.ndarray],
    *,
    title: str = "Trajectory",
    path: str | None = None,
) -> tuple[Figure, Axes]:
    """Plot the mean value of a stored trajectory."""
    plt = _plt()
    values = [
        float(np.mean(np.asarray(frame, dtype=float))) for frame in trajectory
    ]
    fig, ax = plt.subplots()
    ax.plot(values)
    ax.set_title(title)
    ax.set_xlabel("Stored step")
    ax.set_ylabel("Mean value")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
    return fig, ax
