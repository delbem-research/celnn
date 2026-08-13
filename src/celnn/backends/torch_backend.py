"""PyTorch backend: the same aggregation, made differentiable.

This backend exists so that ``celnn.core.dynamics`` -- the single
definition of

    dx/dt = -x + A * y(x) + B * u + z

-- can be evaluated on tensors without being rewritten. It inherits the
stencil algorithm from :class:`~celnn.backends.stencil.StencilBackend` and
supplies only the array primitives, so the differentiable path cannot drift
away from the reference one.

Two capabilities go beyond the NumPy contract, both needed by multichannel
cellular language models:

* **Batch axes.** Leading axes are carried through untouched.
* **Diagonal templates.** A ``(span, channels)`` template applies one weight
  vector per offset, so each channel evolves under its own scalar CelNN. This
  is a deliberate restriction, not an oversight: channels never mix inside the
  dynamics, and anything needing cross-channel computation must add an explicit
  mixing term outside this backend.

With ``spatial_ndim=1`` and no channel axis, the contract collapses onto the
NumPy one exactly, which is what makes the classical case a strict special
case of the differentiable one.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from ..core.boundary import normalize_boundary_mode, pad_kwargs
from ..core.exceptions import BackendError
from .stencil import StencilBackend


class TorchBackend(StencilBackend):
    """Backend implementation for PyTorch tensors."""

    name = "torch"

    def __init__(self, spatial_ndim: int = 1, *, causal: bool = False) -> None:
        if spatial_ndim < 1:
            raise BackendError("spatial_ndim must be at least 1.")
        self.torch = self._import_torch()
        self.spatial_ndim = spatial_ndim
        self.causal = bool(causal)

    @staticmethod
    def _import_torch() -> Any:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - environment branch
            raise BackendError(
                "The torch backend requires PyTorch. Install it with "
                "`pip install celnn[torch]`."
            ) from exc
        return torch

    @classmethod
    def is_available(cls) -> bool:
        """Return whether PyTorch can be imported."""
        try:
            cls._import_torch()
        except BackendError:
            return False
        return True

    # ------------------------------------------------------------------
    # Stencil primitives
    # ------------------------------------------------------------------
    def _prepare(self, value: Any) -> Any:
        if isinstance(value, self.torch.Tensor):
            return value
        return self.torch.as_tensor(np.asarray(value, dtype=float))

    def _zeros_like(self, array: Any) -> Any:
        return self.torch.zeros_like(array)

    def _kernel_axes(self, array: Any, kernel: Any) -> tuple[int, ...]:
        """Locate spatial axes, allowing batch and channel axes."""
        channel_axes = kernel.ndim - self.spatial_ndim
        if channel_axes < 0:
            raise BackendError(
                "Template rank is below the configured spatial rank. "
                f"Got weights ndim={kernel.ndim}, "
                f"spatial_ndim={self.spatial_ndim}."
            )

        last = array.ndim - channel_axes - 1
        first = last - self.spatial_ndim + 1
        if first < 0:
            raise BackendError(
                "Input and template dimensionality must match "
                "for local aggregation. "
                f"Got array ndim={array.ndim}, weights ndim={kernel.ndim}."
            )
        return tuple(range(first, last + 1))

    def _pad(
        self,
        array: Any,
        axes: Sequence[int],
        radii: Sequence[int],
        *,
        mode: str,
        cval: float,
    ) -> Any:
        """Pad one axis at a time, reusing NumPy's own boundary semantics.

        For every mode but ``constant`` the padded result is a gather from the
        original array, and the gather indices are obtained by padding
        ``arange(n)`` with NumPy itself. That makes agreement with the
        reference backend true by construction rather than by careful reading
        of two libraries' documentation -- which matters here, because NumPy
        and SciPy already disagree on what "reflect" means.
        """
        torch = self.torch
        normalized = normalize_boundary_mode(mode)
        padded = array

        for position, axis in enumerate(axes):
            radius = radii[position]
            if radius == 0:
                continue

            if normalized == "constant":
                shape = list(padded.shape)
                shape[axis] = radius
                block = torch.full(
                    shape,
                    float(cval),
                    dtype=padded.dtype,
                    device=padded.device,
                )
                padded = torch.cat([block, padded, block], dim=axis)
                continue

            length = padded.shape[axis]
            gather = np.pad(
                np.arange(length),
                (radius, radius),
                **pad_kwargs(normalized, 0.0),
            )
            index = torch.as_tensor(
                gather, dtype=torch.long, device=padded.device
            )
            padded = torch.index_select(padded, axis, index)

        return padded

    def _stencil_sum(
        self,
        array: Any,
        kernel: Any,
        axes: Sequence[int],
        *,
        mode: str,
        cval: float,
    ) -> Any:
        """Apply a left-looking stencil when causal mode is requested."""
        if not self.causal:
            return super()._stencil_sum(
                array, kernel, axes, mode=mode, cval=cval
            )
        if len(axes) != 1:
            raise BackendError(
                "Causal aggregation currently supports one spatial axis."
            )

        span = kernel.shape[0]
        padded = self._pad_one_sided(
            array, axes[0], span - 1, mode=mode, cval=cval
        )
        result = self._zeros_like(array)
        for offset in range(span):
            window = self._window(padded, array, axes, (offset,))
            result = result + kernel[offset] * window
        return result

    def _pad_one_sided(
        self,
        array: Any,
        axis: int,
        left: int,
        *,
        mode: str,
        cval: float,
    ) -> Any:
        """Pad only the past side of one spatial axis."""
        if left == 0:
            return array
        normalized = normalize_boundary_mode(mode)
        if normalized == "constant":
            shape = list(array.shape)
            shape[axis] = left
            block = self.torch.full(
                shape,
                float(cval),
                dtype=array.dtype,
                device=array.device,
            )
            return self.torch.cat([block, array], dim=axis)

        gather = np.pad(
            np.arange(array.shape[axis]),
            (left, 0),
            **pad_kwargs(normalized, 0.0),
        )
        index = self.torch.as_tensor(
            gather, dtype=self.torch.long, device=array.device
        )
        return self.torch.index_select(array, axis, index)
