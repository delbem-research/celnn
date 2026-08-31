"""One-dimensional signal helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ..core.exceptions import OptionalDependencyError
from ..utils.doc import optional_dependency_message

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


def normalize_signal(
    signal: np.ndarray, *, lower: float = -1.0, upper: float = 1.0
) -> np.ndarray:
    """Normalize a signal to a target interval."""
    array = np.asarray(signal, dtype=float)
    minimum = float(array.min())
    maximum = float(array.max())
    if np.isclose(minimum, maximum):
        return np.full_like(
            array, fill_value=(lower + upper) / 2.0, dtype=float
        )
    scaled = (array - minimum) / (maximum - minimum)
    return lower + (upper - lower) * scaled


def generate_sine_wave(
    *, samples: int = 512, cycles: float = 4.0, amplitude: float = 1.0
) -> np.ndarray:
    """Generate a sine wave."""
    x = np.linspace(0.0, 2.0 * np.pi * cycles, samples)
    return amplitude * np.sin(x)


def generate_noisy_sine(
    *,
    samples: int = 512,
    cycles: float = 4.0,
    amplitude: float = 1.0,
    noise_scale: float = 0.1,
    seed: int | None = 0,
) -> np.ndarray:
    """Generate a sine wave with additive Gaussian noise."""
    rng = np.random.default_rng(seed)
    return generate_sine_wave(
        samples=samples, cycles=cycles, amplitude=amplitude
    ) + rng.normal(scale=noise_scale, size=samples)


def _plt() -> Any:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - optional dependency branch
        raise OptionalDependencyError(
            optional_dependency_message("matplotlib", "viz", "Signal plotting")
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


def plot_signal_comparison(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    title: str = "Signal comparison",
    path: str | None = None,
) -> tuple[Figure, Axes]:
    """Plot a reference and a processed signal."""
    plt = _plt()
    fig, ax = plt.subplots()
    ax.plot(np.asarray(reference, dtype=float), label="reference")
    ax.plot(np.asarray(candidate, dtype=float), label="candidate")
    ax.set_title(title)
    ax.set_xlabel("Index")
    ax.set_ylabel("Value")
    ax.legend()
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
    return fig, ax
