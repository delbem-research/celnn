"""Boundary-condition utilities."""

from __future__ import annotations

from typing import Any

from .exceptions import CelNNError

VALID_BOUNDARY_MODES = ("constant", "wrap", "reflect", "nearest", "mirror")

_NUMPY_PAD_MODES = {
    "constant": "constant",
    "wrap": "wrap",
    "reflect": "reflect",
    "nearest": "edge",
    "mirror": "symmetric",
}


def normalize_boundary_mode(mode: str) -> str:
    """Normalize and validate a boundary mode."""
    normalized = mode.lower().strip()
    if normalized not in VALID_BOUNDARY_MODES:
        valid = ", ".join(VALID_BOUNDARY_MODES)
        raise CelNNError(
            f"Unsupported boundary mode '{mode}'. Expected one of: {valid}."
        )
    return normalized


def numpy_pad_mode(mode: str) -> str:
    """Map a public boundary mode to the equivalent NumPy pad mode."""
    normalized = normalize_boundary_mode(mode)
    return _NUMPY_PAD_MODES[normalized]


def pad_kwargs(mode: str, boundary_value: float) -> dict[str, Any]:
    """Return keyword arguments for ``numpy.pad``."""
    normalized = normalize_boundary_mode(mode)
    if normalized == "constant":
        return {"mode": "constant", "constant_values": boundary_value}
    return {"mode": numpy_pad_mode(normalized)}


def scipy_mode(mode: str) -> str:
    """Return the SciPy ``ndimage`` mode string."""
    normalized = normalize_boundary_mode(mode)
    # NumPy and SciPy invert the names of the two reflection modes.
    if normalized == "reflect":
        return "mirror"
    if normalized == "mirror":
        return "reflect"
    return normalized
