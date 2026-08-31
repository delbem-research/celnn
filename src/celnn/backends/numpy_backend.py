"""NumPy backend with optional SciPy acceleration."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from ..core.boundary import pad_kwargs, scipy_mode
from .stencil import StencilBackend

try:  # pragma: no cover - optional dependency branch
    from scipy.ndimage import convolve as scipy_convolve
except ImportError:  # pragma: no cover - optional dependency branch
    scipy_convolve = None


class NumPyBackend(StencilBackend):
    """Backend implementation for NumPy arrays."""

    name = "numpy"

    def _prepare(self, value: Any) -> np.ndarray:
        return np.asarray(value)

    def _zeros_like(self, array: np.ndarray) -> np.ndarray:
        return np.zeros_like(array)

    def _pad(
        self,
        array: np.ndarray,
        axes: Sequence[int],
        radii: Sequence[int],
        *,
        mode: str,
        cval: float,
    ) -> np.ndarray:
        widths = [(0, 0)] * array.ndim
        for position, axis in enumerate(axes):
            widths[axis] = (radii[position], radii[position])
        return np.pad(array, widths, **pad_kwargs(mode, cval))

    def _fast_path(
        self,
        array: np.ndarray,
        kernel: np.ndarray,
        *,
        mode: str,
        cval: float,
    ) -> np.ndarray | None:
        """Delegate to SciPy's optimised convolution when it is available."""
        if scipy_convolve is None:
            return None
        flipped = np.flip(kernel)
        return np.asarray(
            scipy_convolve(
                array, flipped, mode=scipy_mode(mode), cval=float(cval)
            ),
            dtype=array.dtype,
        )


NUMPY_BACKEND = NumPyBackend()


def get_default_backend() -> NumPyBackend:
    """Return the default backend instance."""
    return NUMPY_BACKEND
