"""Shared stencil algorithm for every array backend.

Applying a CelNN template is always the same computation: pad the field, then
accumulate one weighted, shifted copy of it per template offset. Only the
array primitives differ between NumPy, CuPy and PyTorch -- how you pad, and how
you allocate an accumulator.

This base class owns the algorithm; subclasses supply the primitives. Keeping
the loop in one place is what lets the differentiable backend inherit the exact
aggregation semantics of the reference one, boundary handling included, rather
than reimplementing them and hoping they agree.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from ..core.exceptions import BackendError


class StencilBackend:
    """Template application by weighted accumulation of shifted windows."""

    name = "stencil"

    # ------------------------------------------------------------------
    # Primitives supplied by subclasses
    # ------------------------------------------------------------------
    def _prepare(self, value: Any) -> Any:
        """Coerce user input into this backend's array type."""
        raise NotImplementedError

    def _pad(
        self,
        array: Any,
        axes: Sequence[int],
        radii: Sequence[int],
        *,
        mode: str,
        cval: float,
    ) -> Any:
        """Pad ``array`` by ``radii`` on each of ``axes``."""
        raise NotImplementedError

    def _zeros_like(self, array: Any) -> Any:
        """Return a zero accumulator shaped like ``array``."""
        raise NotImplementedError

    def _fast_path(
        self, array: Any, kernel: Any, *, mode: str, cval: float
    ) -> Any | None:
        """Optional accelerated implementation; ``None`` falls back."""
        return None

    def _finalize(self, result: Any) -> Any:
        """Convert the accumulator to the type callers expect.

        Device backends override this to bring the result back to the host;
        the public API of the library returns NumPy arrays regardless of where
        the arithmetic happened.
        """
        return result

    # ------------------------------------------------------------------
    # Shared algorithm
    # ------------------------------------------------------------------
    def _kernel_axes(self, array: Any, kernel: Any) -> tuple[int, ...]:
        """Return the array axes the kernel slides over.

        The default contract is the historical one: the template has exactly
        one axis per array axis. Backends that carry batch or channel axes
        override this.
        """
        if array.ndim != kernel.ndim:
            raise BackendError(
                "Input and template dimensionality must match "
                "for local aggregation. "
                f"Got array ndim={array.ndim}, weights ndim={kernel.ndim}."
            )
        return tuple(range(array.ndim))

    def aggregate_local(
        self,
        values: Any,
        weights: Any,
        *,
        mode: str,
        cval: float = 0.0,
    ) -> Any:
        """Aggregate local neighborhoods using stencil-aligned weights."""
        array = self._prepare(values)
        kernel = self._prepare(weights)
        axes = self._kernel_axes(array, kernel)

        accelerated = self._fast_path(array, kernel, mode=mode, cval=cval)
        if accelerated is not None:
            return self._finalize(accelerated)

        return self._finalize(
            self._stencil_sum(array, kernel, axes, mode=mode, cval=cval)
        )

    def _stencil_sum(
        self,
        array: Any,
        kernel: Any,
        axes: Sequence[int],
        *,
        mode: str,
        cval: float,
    ) -> Any:
        """Accumulate ``sum_j kernel[j] * shift_j(array)`` over the stencil."""
        spans = tuple(kernel.shape[: len(axes)])
        padded = self._pad_for_stencil(
            array, axes, spans, mode=mode, cval=cval
        )

        result = self._zeros_like(array)
        for offset in np.ndindex(*spans):
            window = self._window(padded, array, axes, offset)
            result = result + kernel[offset] * window
        return result

    def _pad_for_stencil(
        self,
        array: Any,
        axes: Sequence[int],
        spans: Sequence[int],
        *,
        mode: str,
        cval: float,
    ) -> Any:
        """Pad the field for the default centered stencil layout."""
        radii = tuple(span // 2 for span in spans)
        return self._pad(array, axes, radii, mode=mode, cval=cval)

    @staticmethod
    def _window(
        padded: Any, array: Any, axes: Sequence[int], offset: Sequence[int]
    ) -> Any:
        """Slice the shifted copy of ``array`` selected by ``offset``."""
        slicer: list[Any] = [slice(None)] * padded.ndim
        for position, axis in enumerate(axes):
            start = offset[position]
            length = array.shape[axis]
            slicer[axis] = slice(start, start + length)
        return padded[tuple(slicer)]
