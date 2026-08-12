"""Integration formulas are shared between the simulation and torch paths."""

from __future__ import annotations

import numpy as np
import pytest

from celnn.core.steppers import STEPPERS, euler_step, semi_implicit_euler_step

torch = pytest.importorskip("torch", reason="PyTorch is optional")


def test_euler_step_is_the_textbook_formula():
    state = np.array([1.0, 2.0])
    got = euler_step(state, 0.5, np.array([2.0, -4.0]))
    np.testing.assert_allclose(got, [2.0, 0.0])


def test_semi_implicit_step_treats_the_leak_implicitly():
    state = np.array([1.0])
    got = semi_implicit_euler_step(state, 1.0, np.array([3.0]))
    np.testing.assert_allclose(got, [2.0])


def test_pure_decay_contracts_geometrically():
    state = np.array([1.0])
    for _ in range(3):
        state = euler_step(state, 0.5, -state)
    np.testing.assert_allclose(state, [0.125])


@pytest.mark.parametrize("name", sorted(STEPPERS))
def test_steppers_are_array_library_agnostic(name):
    stepper = STEPPERS[name]
    rate = np.array([0.7, -1.3, 0.2])
    state = np.array([1.0, 2.0, 3.0])

    expected = stepper(state, 0.25, rate)
    got = stepper(
        torch.tensor(state, dtype=torch.float64),
        0.25,
        torch.tensor(rate, dtype=torch.float64),
    )

    assert isinstance(got, torch.Tensor)
    np.testing.assert_allclose(got.numpy(), expected, rtol=1e-12, atol=1e-12)


def test_steppers_are_differentiable():
    state = torch.ones(4, dtype=torch.float64, requires_grad=True)
    euler_step(state, 0.5, -state).sum().backward()
    assert state.grad is not None
    np.testing.assert_allclose(state.grad.numpy(), np.full(4, 0.5))
