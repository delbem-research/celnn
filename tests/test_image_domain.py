import numpy as np
import pytest

from celnn.domains.image import (
    denormalize_image,
    load_grayscale,
    normalize_image,
    save_grayscale,
)

pytest.importorskip("PIL")


def test_image_roundtrip(tmp_path):
    image = np.linspace(-1.0, 1.0, 64).reshape(8, 8)
    path = tmp_path / "demo.png"
    save_grayscale(image, path)
    restored = load_grayscale(path)
    assert restored.shape == image.shape
    assert np.all(restored <= 1.0)
    assert np.all(restored >= -1.0)


def test_image_normalization_helpers():
    image = np.array([[0, 127, 255]], dtype=np.uint8)
    normalized = normalize_image(image)
    restored = denormalize_image(normalized)
    assert restored.dtype == np.uint8
    assert restored.shape == image.shape


def test_unit_interval_image_uses_full_celnn_range():
    image = np.array([0.0, 0.5, 1.0])

    normalized = normalize_image(image)

    assert np.array_equal(normalized, np.array([-1.0, 0.0, 1.0]))
