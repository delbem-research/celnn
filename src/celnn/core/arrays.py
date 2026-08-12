"""Array-namespace dispatch.

The cellular dynamics are the same mathematics whether they are evaluated with
NumPy, CuPy or PyTorch. What differs is only the array library that supplies
``tanh``, ``exp`` and friends. This module resolves that library from the array
itself, so a single implementation of the dynamics can serve all three instead
of being written once per backend.

Two rules keep the dispatch honest:

* NumPy-shaped inputs (lists, scalars, ndarrays) are coerced to float arrays,
  preserving the historical behaviour of the library.
* Foreign arrays -- torch tensors, CuPy arrays -- are passed through untouched.
  Coercing a grad-tracking tensor through ``numpy.asarray`` raises, and even
  where it succeeds it severs the autograd graph, which would silently defeat
  the entire differentiable path.
"""

from __future__ import annotations

from types import ModuleType
from typing import Any

import numpy as np

__all__ = ["array_namespace", "as_float_array", "is_foreign_array"]

_FOREIGN_MODULES = ("torch", "cupy")


def _root_module(value: Any) -> str:
    """Return the top-level module name of ``value``'s type."""
    return type(value).__module__.partition(".")[0]


def is_foreign_array(value: Any) -> bool:
    """Return ``True`` for arrays owned by a non-NumPy array library."""
    return _root_module(value) in _FOREIGN_MODULES


def array_namespace(value: Any) -> ModuleType:
    """Return the array library that should evaluate operations on ``value``.

    Falls back to NumPy for lists, scalars and anything else unrecognised, so
    callers never need to special-case plain Python input.
    """
    root = _root_module(value)
    if root in _FOREIGN_MODULES:
        import importlib

        return importlib.import_module(root)
    return np


def as_float_array(value: Any) -> Any:
    """Coerce NumPy-shaped input; pass foreign arrays through untouched."""
    if is_foreign_array(value):
        return value
    return np.asarray(value, dtype=float)
