"""CuPy backend for GPU local aggregation."""

from __future__ import annotations

import os
import site
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..core.boundary import pad_kwargs, scipy_mode
from ..core.exceptions import BackendError
from .stencil import StencilBackend


class CuPyBackend(StencilBackend):
    """Backend implementation for CuPy/CUDA arrays.

    The public `celnn` API still returns NumPy arrays. This backend moves
    stencil aggregation to the GPU and converts the result back to NumPy at
    the backend boundary.
    """

    name = "cupy"

    def __init__(self) -> None:
        self.cp = self._import_cupy()

    @classmethod
    def is_available(cls) -> bool:
        """Return whether CuPy can execute a small CUDA operation."""
        try:
            cls._configure_cuda_path()
            import cupy as cp

            if cp.cuda.runtime.getDeviceCount() <= 0:
                return False
            test = cp.arange(4, dtype=cp.float32)
            float(cp.sum(test).get())
        except Exception:
            return False
        return True

    def _prepare(self, value: Any) -> Any:
        return self.cp.asarray(value, dtype=self.cp.float64)

    def _zeros_like(self, array: Any) -> Any:
        return self.cp.zeros_like(array, dtype=self.cp.float64)

    def _pad(
        self,
        array: Any,
        axes: Sequence[int],
        radii: Sequence[int],
        *,
        mode: str,
        cval: float,
    ) -> Any:
        widths = [(0, 0)] * array.ndim
        for position, axis in enumerate(axes):
            widths[axis] = (radii[position], radii[position])
        return self.cp.pad(array, widths, **pad_kwargs(mode, cval))

    def _fast_path(
        self, array: Any, kernel: Any, *, mode: str, cval: float
    ) -> Any | None:
        """Use cupyx convolution for ND, where the Python loop is costly.

        1D and 2D keep the shared stencil, which the stubbed-runtime tests
        compare against the NumPy backend mode by mode.
        """
        if array.ndim <= 2:
            return None
        try:
            from cupyx.scipy.ndimage import convolve
        except ImportError:  # pragma: no cover - optional branch
            return None
        return convolve(
            array,
            self.cp.flip(kernel),
            mode=scipy_mode(mode),
            cval=float(cval),
        )

    def _finalize(self, result: Any) -> np.ndarray:
        """Bring the result back to the host; the public API returns NumPy."""
        return np.asarray(self.cp.asnumpy(result), dtype=float)

    @classmethod
    def _import_cupy(cls) -> Any:
        cls._configure_cuda_path()
        try:
            import cupy as cp
        except ImportError as exc:  # pragma: no cover - environment branch
            raise BackendError(
                "The CuPy backend requires CuPy. Install GPU support with "
                "`pip install celnn[gpu]`."
            ) from exc
        try:
            if cp.cuda.runtime.getDeviceCount() <= 0:
                raise BackendError("No CUDA device is visible to CuPy.")
            test = cp.arange(4, dtype=cp.float32)
            float(cp.sum(test).get())
        except BackendError:
            raise
        except Exception as exc:
            raise BackendError(
                "CuPy is installed, but CUDA execution failed. Install a "
                "matching CUDA runtime or use device='auto'/'cpu'."
            ) from exc
        return cp

    @staticmethod
    def _configure_cuda_path() -> None:
        if os.environ.get("CUDA_PATH"):
            return
        for package_dir in site.getsitepackages():
            nvidia_dir = Path(package_dir) / "nvidia"
            runtime_candidate = nvidia_dir / "cuda_runtime"
            if (runtime_candidate / "include" / "cuda.h").exists():
                os.environ["CUDA_PATH"] = str(runtime_candidate)
                return
            nvrtc_candidate = nvidia_dir / "cuda_nvrtc"
            if (
                (nvrtc_candidate / "include" / "cuda.h").exists()
                and any((nvrtc_candidate / "lib").glob("libnvrtc.so*"))
            ):
                os.environ["CUDA_PATH"] = str(nvrtc_candidate)
                return

    def _aggregate_1d(self, array, kernel, *, mode: str, cval: float):
        radius = kernel.shape[0] // 2
        padded = self.cp.pad(
            array,
            (radius, radius),
            **self._pad_kwargs(mode, cval),
        )
        result = self.cp.zeros_like(array, dtype=self.cp.float64)
        for index, weight in enumerate(kernel):
            result += weight * padded[index: index + array.shape[0]]
        return result

    def _aggregate_2d(self, array, kernel, *, mode: str, cval: float):
        pad_y = kernel.shape[0] // 2
        pad_x = kernel.shape[1] // 2
        padded = self.cp.pad(
            array,
            ((pad_y, pad_y), (pad_x, pad_x)),
            **self._pad_kwargs(mode, cval),
        )
        result = self.cp.zeros_like(array, dtype=self.cp.float64)
        height, width = array.shape
        for row in range(kernel.shape[0]):
            row_slice = slice(row, row + height)
            for col in range(kernel.shape[1]):
                col_slice = slice(col, col + width)
                result += kernel[row, col] * padded[row_slice, col_slice]
        return result

    def _aggregate_nd(self, array, kernel, *, mode: str, cval: float):
        try:
            from cupyx.scipy.ndimage import convolve
        except ImportError as exc:  # pragma: no cover - optional branch
            raise BackendError(
                "CuPy ND aggregation requires cupyx.scipy.ndimage. "
                "Use 1D/2D templates or device='cpu'."
            ) from exc
        return convolve(
            array,
            self.cp.flip(kernel),
            mode=self._scipy_mode(mode),
            cval=float(cval),
        )
