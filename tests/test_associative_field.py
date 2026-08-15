"""Normalized associative field state remains explicit and differentiable."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from celnn.associative_field import (  # noqa: E402
    AssociativeFieldState,
    NormalizedDeltaHebbianField,
)


def make_field(**overrides):
    values = {
        "key_size": 3,
        "value_size": 2,
        "learning_rate": 0.5,
        "retention": 1.0,
    }
    values.update(overrides)
    return NormalizedDeltaHebbianField(**values)


def test_state_is_explicit_local_resettable_and_movable():
    field = make_field()
    state = field.new_state(2, 5, like=torch.ones(1))

    assert state.memory.shape == (2, 5, 2, 3)
    assert state.normalizer.shape == (2, 5, 3)
    assert state.updates == 0
    assert torch.count_nonzero(state.memory) == 0
    assert state.reset().normalizer.sum() == 0
    assert state.to(dtype=torch.float64).memory.dtype == torch.float64


def test_positive_feature_map_keeps_denominator_non_negative():
    field = make_field()
    state = field.new_state(1, 2, like=torch.ones(1))
    key = torch.randn(1, 2, 3)
    value = torch.randn(1, 2, 2)
    written = field.write(state, key, value)
    features = field.feature_map(torch.randn(1, 2, 3))
    denominator = torch.einsum(
        "bck,bck->bc", written.normalizer, features
    )

    assert torch.all(field.feature_map(key) > 0)
    assert torch.all(denominator >= 0)


def test_full_rate_write_retrieves_written_value_for_same_key():
    field = make_field(learning_rate=1.0)
    state = field.new_state(1, 2, like=torch.ones(1))
    key = torch.tensor([[[1.0, -0.5, 0.2], [-0.2, 0.3, 1.1]]])
    value = torch.tensor([[[0.4, -0.7], [0.1, 0.8]]])

    written = field.write(state, key, value)
    retrieved = field.read(written, key)

    torch.testing.assert_close(retrieved, value, atol=1e-5, rtol=1e-5)


def test_cells_update_independently_and_mask_preserves_inactive_cells():
    field = make_field()
    state = field.new_state(1, 3, like=torch.ones(1))
    key = torch.randn(1, 3, 3)
    value = torch.randn(1, 3, 2)

    written = field.write(
        state,
        key,
        value,
        mask=torch.tensor([[True, False, False]]),
    )

    assert torch.count_nonzero(written.memory[:, :1]) > 0
    assert torch.count_nonzero(written.memory[:, 1:]) == 0
    assert torch.count_nonzero(written.normalizer[:, 1:]) == 0


def test_delta_write_reduces_existing_association_error():
    field = make_field(learning_rate=0.2)
    state = field.new_state(1, 1, like=torch.ones(1))
    key = torch.randn(1, 1, 3)
    value = torch.tensor([[[0.8, -0.4]]])

    once = field.write(state, key, value)
    before = (field.read(once, key) - value).square().sum()
    twice = field.write(once, key, value)
    after = (field.read(twice, key) - value).square().sum()

    assert after < before
    torch.testing.assert_close(after, 0.64 * before, atol=1e-5, rtol=1e-5)


def test_rates_can_be_controlled_per_cell_and_receive_gradients():
    field = make_field()
    state = field.new_state(1, 2, like=torch.ones(1))
    key = torch.randn(1, 2, 3, requires_grad=True)
    value = torch.randn(1, 2, 2, requires_grad=True)
    rate = torch.tensor([[0.1, 0.3]], requires_grad=True)
    keep = torch.tensor([[0.9, 0.8]], requires_grad=True)

    written = field.write(
        state, key, value, learning_rate=rate, retention=keep
    )
    field.read(written, key).square().sum().backward()

    for tensor in (key, value, rate, keep):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()


def test_state_rejects_mismatched_normalizer_shape():
    with pytest.raises(ValueError, match="normalizer"):
        AssociativeFieldState(torch.zeros(1, 2, 3, 4), torch.zeros(1, 3, 4))
