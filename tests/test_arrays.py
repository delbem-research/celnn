"""Array-namespace dispatch shared by simulation and differentiable paths."""

from __future__ import annotations

import numpy as np
import pytest

from celnn.core.arrays import (
    array_namespace,
    as_float_array,
    is_foreign_array,
)

torch = pytest.importorskip("torch", reason="PyTorch is optional")


def test_namespace_of_numpy_array_is_numpy():
    assert array_namespace(np.zeros(3)) is np


def test_namespace_of_plain_python_is_numpy():
    assert array_namespace([1.0, 2.0]) is np
    assert array_namespace(3.0) is np


def test_namespace_of_torch_tensor_is_torch():
    assert array_namespace(torch.zeros(3)) is torch


def test_numpy_inputs_are_coerced_to_float_arrays():
    coerced = as_float_array([1, 2, 3])
    assert isinstance(coerced, np.ndarray)
    assert coerced.dtype == np.dtype(float)


def test_torch_tensors_pass_through_untouched():
    tensor = torch.ones(3, requires_grad=True)
    assert as_float_array(tensor) is tensor


def test_coercion_would_destroy_the_autograd_graph():
    """Why as_float_array exists: np.asarray refuses a grad-tracking tensor."""
    tensor = torch.ones(3, requires_grad=True)
    with pytest.raises(RuntimeError):
        np.asarray(tensor, dtype=float)


def test_is_foreign_array_distinguishes_tensors_from_ndarrays():
    assert is_foreign_array(torch.zeros(2)) is True
    assert is_foreign_array(np.zeros(2)) is False
    assert is_foreign_array([0.0, 1.0]) is False
