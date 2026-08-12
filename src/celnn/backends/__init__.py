"""Backend utilities."""

from typing import Any

from .cupy_backend import CuPyBackend
from .numpy_backend import NUMPY_BACKEND, NumPyBackend, get_default_backend
from .protocol import ArrayBackend
from .stencil import StencilBackend


def __getattr__(name: str) -> Any:
    """Expose TorchBackend lazily so PyTorch stays optional."""
    if name == "TorchBackend":
        from .torch_backend import TorchBackend

        return TorchBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def get_backend(device: str = "cpu") -> ArrayBackend:
    """Return a backend for a requested execution device."""
    normalized = device.lower().strip()
    if normalized == "cpu":
        return NUMPY_BACKEND
    if normalized in {"gpu", "cuda"}:
        return CuPyBackend()
    if normalized == "auto":
        if CuPyBackend.is_available():
            return CuPyBackend()
        return NUMPY_BACKEND
    raise ValueError("device must be one of: cpu, gpu, cuda, auto.")


__all__ = [
    "ArrayBackend",
    "CuPyBackend",
    "NumPyBackend",
    "NUMPY_BACKEND",
    "StencilBackend",
    "TorchBackend",
    "get_backend",
    "get_default_backend",
]
