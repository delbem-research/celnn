"""Optional Pillow-based image helpers."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from ..core.exceptions import OptionalDependencyError
from ..utils.doc import optional_dependency_message

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage


def _image_module() -> Any:
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - optional dependency branch
        raise OptionalDependencyError(
            optional_dependency_message("pillow", "image", "Image support")
        ) from exc
    return Image


def normalize_image(image: Any) -> np.ndarray:
    """Normalize image-like data to floating values in ``[-1, 1]``."""
    array = np.asarray(image, dtype=float)
    if array.size == 0:
        return array.astype(float)
    if array.min() >= -1.0 and array.max() <= 1.0:
        return array.astype(float, copy=False)
    if array.min() >= 0.0 and array.max() <= 1.0:
        return (2.0 * array) - 1.0
    return (array / 127.5) - 1.0


def denormalize_image(image: Any) -> np.ndarray:
    """Map ``[-1, 1]`` data to ``uint8`` grayscale."""
    array = np.asarray(image, dtype=float)
    clipped = np.clip(array, -1.0, 1.0)
    return np.round((clipped + 1.0) * 127.5).astype(np.uint8)


def image_to_array(image: Any) -> np.ndarray:
    """Convert a PIL image or image-like value to a normalized NumPy array."""
    return normalize_image(np.asarray(image))


def array_to_image(array: Any) -> PILImage:
    """Convert a normalized array to a grayscale PIL image."""
    Image = _image_module()
    grayscale = denormalize_image(array)
    return Image.fromarray(grayscale, mode="L")


def load_grayscale(
    path: str | Path, *, dtype: Any | None = None
) -> np.ndarray:
    """Load a grayscale image and normalize it to ``[-1, 1]``."""
    Image = _image_module()
    image = Image.open(path).convert("L")
    array = normalize_image(np.asarray(image))
    if dtype is not None:
        return array.astype(dtype)
    return array


def save_grayscale(array: Any, path: str | Path) -> Path:
    """Save a normalized array as a grayscale image."""
    image = array_to_image(array)
    target = Path(path)
    image.save(target)
    return target
