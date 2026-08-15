"""Delta-Hebbian associative memory is local, explicit, and reusable."""

from __future__ import annotations

import pytest

import celnn

torch = pytest.importorskip("torch", reason="PyTorch is optional")


def test_associative_api_is_lazy_and_public():
    from celnn import (
        AssociativeMemoryState,
        DeltaHebbianMemory,
        DeltaHebbianRule,
    )

    assert AssociativeMemoryState is celnn.AssociativeMemoryState
    assert DeltaHebbianMemory is celnn.DeltaHebbianMemory
    assert DeltaHebbianRule is celnn.DeltaHebbianRule


def test_delta_hebb_writes_and_retrieves_key_value_association():
    memory = celnn.DeltaHebbianMemory(
        2,
        2,
        celnn.DeltaHebbianRule(learning_rate=1.0, retention=1.0),
    )
    state = memory.new_state(1, like=torch.zeros(1))
    key = torch.tensor([[3.0, 0.0]])
    value = torch.tensor([[2.0, -1.0]])

    written = memory.write(state, key, value)
    retrieved = memory.read(written, key)

    torch.testing.assert_close(retrieved, value)
    assert state.updates == 0
    assert written.updates == 1


def test_delta_rule_corrects_instead_of_reinforcing_old_value():
    memory = celnn.DeltaHebbianMemory(
        2,
        1,
        celnn.DeltaHebbianRule(learning_rate=1.0, retention=1.0),
    )
    state = memory.new_state(1, like=torch.zeros(1))
    key = torch.tensor([[1.0, 0.0]])
    first = memory.write(state, key, torch.tensor([[2.0]]))
    corrected = memory.write(first, key, torch.tensor([[5.0]]))

    torch.testing.assert_close(
        memory.read(corrected, key), torch.tensor([[5.0]])
    )


def test_dynamic_write_and_retention_gates_are_batch_local():
    memory = celnn.DeltaHebbianMemory(1, 1)
    state = celnn.AssociativeMemoryState(torch.ones(2, 1, 1))
    written = memory.write(
        state,
        torch.ones(2, 1),
        torch.tensor([[3.0], [5.0]]),
        learning_rate=torch.tensor([0.0, 1.0]),
        retention=torch.tensor([0.5, 1.0]),
    )

    torch.testing.assert_close(
        written.memory[:, 0, 0], torch.tensor([0.5, 5.0])
    )


def test_differentiable_write_propagates_through_memory():
    memory = celnn.DeltaHebbianMemory(2, 2, detach_updates=False)
    state = memory.new_state(1, like=torch.zeros(1))
    key = torch.randn(1, 2, requires_grad=True)
    value = torch.randn(1, 2, requires_grad=True)

    retrieved = memory.read(memory.write(state, key, value), key)
    retrieved.sum().backward()

    assert key.grad is not None
    assert value.grad is not None


def test_state_reset_detach_cast_and_memory_limit():
    memory = celnn.DeltaHebbianMemory(
        1,
        1,
        celnn.DeltaHebbianRule(learning_rate=10.0),
        memory_limit=0.25,
    )
    state = memory.new_state(2, like=torch.ones(1, requires_grad=True))
    written = memory.write(state, torch.ones(2, 1), torch.ones(2, 1))

    assert written.memory.abs().max() == 0.25
    assert written.reset().updates == 0
    assert written.detach().memory.grad_fn is None
    assert written.to(dtype=torch.float64).memory.dtype == torch.float64
