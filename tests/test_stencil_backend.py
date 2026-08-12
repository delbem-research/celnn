"""The stencil algorithm is shared; only its array primitives differ."""

from __future__ import annotations

import numpy as np
import pytest

from celnn.backends import numpy_backend as numpy_backend_module
from celnn.backends.numpy_backend import NumPyBackend
from celnn.backends.stencil import StencilBackend
from celnn.core.boundary import VALID_BOUNDARY_MODES
from celnn.core.exceptions import BackendError


@pytest.fixture
def without_scipy(monkeypatch):
    """Force the pure-Python stencil path."""
    monkeypatch.setattr(numpy_backend_module, "scipy_convolve", None)


def _both_paths(monkeypatch, array, kernel, mode):
    """Return (scipy result, shared-stencil result) for the same input."""
    if numpy_backend_module.scipy_convolve is None:
        pytest.skip("SciPy is not installed")
    accelerated = NumPyBackend().aggregate_local(
        array, kernel, mode=mode, cval=0.0
    )
    monkeypatch.setattr(numpy_backend_module, "scipy_convolve", None)
    fallback = NumPyBackend().aggregate_local(
        array, kernel, mode=mode, cval=0.0
    )
    return accelerated, fallback


def test_numpy_backend_is_a_stencil_backend():
    assert isinstance(NumPyBackend(), StencilBackend)


@pytest.mark.parametrize("mode", VALID_BOUNDARY_MODES)
def test_shared_stencil_matches_the_scipy_fast_path_1d(mode, monkeypatch):
    rng = np.random.default_rng(0)
    array = rng.normal(size=17)
    kernel = np.array([0.2, -0.5, 1.0, -0.5, 0.2])

    accelerated, fallback = _both_paths(monkeypatch, array, kernel, mode)
    np.testing.assert_allclose(fallback, accelerated, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("mode", VALID_BOUNDARY_MODES)
def test_shared_stencil_matches_the_scipy_fast_path_2d(mode, monkeypatch):
    rng = np.random.default_rng(1)
    array = rng.normal(size=(9, 11))
    kernel = np.array([[0.0, 0.1, 0.0], [0.1, 1.0, 0.1], [0.0, 0.1, 0.0]])

    accelerated, fallback = _both_paths(monkeypatch, array, kernel, mode)
    np.testing.assert_allclose(fallback, accelerated, rtol=1e-12, atol=1e-12)


def test_constant_mode_uses_the_fill_value(without_scipy):
    array = np.ones(4)
    kernel = np.array([1.0, 0.0, 0.0])  # picks the left neighbour
    got = NumPyBackend().aggregate_local(
        array, kernel, mode="constant", cval=7.0
    )
    assert got[0] == pytest.approx(7.0)


def test_three_dimensional_fallback_now_works(without_scipy):
    """The shared loop generalises; the old fallback refused ndim > 2."""
    array = np.arange(2 * 3 * 4, dtype=float).reshape(2, 3, 4)
    kernel = np.zeros((3, 3, 3))
    kernel[1, 1, 1] = 1.0  # identity stencil
    got = NumPyBackend().aggregate_local(array, kernel, mode="constant")
    np.testing.assert_allclose(got, array)


def test_rank_mismatch_still_raises():
    with pytest.raises(BackendError, match="dimensionality"):
        NumPyBackend().aggregate_local(
            np.ones(5), np.ones((3, 3)), mode="constant"
        )
