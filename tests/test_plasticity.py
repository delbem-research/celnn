"""Fast-weight plasticity is modular, functional, and batch-local."""

from __future__ import annotations

import pytest

import celnn

torch = pytest.importorskip("torch", reason="PyTorch is optional")


def test_plasticity_api_is_lazy_and_public():
    from celnn import HebbianRule, OjaRule, Plasticity, PlasticityState

    assert HebbianRule is celnn.HebbianRule
    assert OjaRule is celnn.OjaRule
    assert Plasticity is celnn.Plasticity
    assert PlasticityState is celnn.PlasticityState


def test_hebbian_rule_matches_outer_product_and_decay():
    rule = celnn.HebbianRule(learning_rate=0.5, decay=0.25)
    memory = torch.ones(1, 2, 3)
    pre = torch.tensor([[1.0, 2.0, 3.0]])
    post = torch.tensor([[4.0, 5.0]])

    actual = rule(pre, post, memory)
    expected = 0.25 * memory + 0.5 * torch.tensor(
        [[[4.0, 8.0, 12.0], [5.0, 10.0, 15.0]]]
    )
    torch.testing.assert_close(actual, expected)


def test_rules_average_sequence_samples_without_mixing_batches():
    rule = celnn.HebbianRule()
    memory = torch.zeros(2, 1, 1)
    pre = torch.tensor([[[1.0], [3.0]], [[10.0], [20.0]]])
    post = torch.ones_like(pre)

    actual = rule(pre, post, memory)

    torch.testing.assert_close(actual[:, 0, 0], torch.tensor([0.02, 0.15]))


def test_oja_rule_subtracts_postsynaptic_stabilization():
    rule = celnn.OjaRule(learning_rate=0.1, decay=1.0)
    memory = torch.tensor([[[2.0]]])
    pre = torch.tensor([[3.0]])
    post = torch.tensor([[2.0]])

    actual = rule(pre, post, memory)

    # H + eta * (y*x - y^2*H) = 2 + .1 * (6 - 8)
    torch.testing.assert_close(actual, torch.tensor([[[1.8]]]))


def test_state_reset_detach_and_cast_are_functional():
    source = torch.ones(2, 3, 4, requires_grad=True)
    state = celnn.PlasticityState(source, updates=7)

    reset = state.reset()
    detached = state.detach()
    converted = state.to(dtype=torch.float64)

    assert reset.updates == 0
    assert torch.count_nonzero(reset.memory) == 0
    assert detached.updates == 7 and not detached.memory.requires_grad
    assert converted.memory.dtype == torch.float64
    assert torch.count_nonzero(state.memory) == source.numel()


def test_plasticity_composes_slow_and_fast_weights():
    plasticity = celnn.Plasticity(celnn.HebbianRule(), alpha=0.5)
    slow = torch.tensor([[1.0, 2.0]])
    state = celnn.PlasticityState(torch.tensor([[[2.0, 4.0]]]))

    effective = plasticity.effective_weight(slow, state)

    torch.testing.assert_close(effective, torch.tensor([[[2.0, 4.0]]]))


def test_updates_can_preserve_or_cut_the_autograd_graph():
    pre = torch.randn(1, 2, requires_grad=True)
    post = torch.randn(1, 3, requires_grad=True)
    weight = torch.zeros(3, 2)

    detached = celnn.Plasticity(
        celnn.HebbianRule(), detach_updates=True
    )
    differentiable = celnn.Plasticity(
        celnn.HebbianRule(), detach_updates=False
    )

    detached_state = detached.update(detached.new_state(1, weight), pre, post)
    graph_state = differentiable.update(
        differentiable.new_state(1, weight), pre, post
    )

    assert not detached_state.memory.requires_grad
    assert graph_state.memory.requires_grad
    graph_state.memory.sum().backward()
    assert pre.grad is not None and post.grad is not None


def test_plastic_linear_owns_no_hidden_conversation_state():
    module = celnn.PlasticLinear(
        2, 1, celnn.Plasticity(celnn.HebbianRule(learning_rate=1.0))
    )
    with torch.no_grad():
        module.linear.weight.zero_()
        module.linear.bias.fill_(1.0)
    state = module.new_state(batch_size=1)

    first, learned = module(torch.tensor([[2.0, 3.0]]), state)
    second, unchanged = module(
        torch.tensor([[2.0, 3.0]]), learned, update=False
    )

    torch.testing.assert_close(first, torch.tensor([[1.0]]))
    torch.testing.assert_close(second, torch.tensor([[14.0]]))
    assert state.updates == 0
    assert learned.updates == 1
    assert unchanged is learned


@pytest.mark.parametrize(
    "rule", [celnn.HebbianRule(), celnn.OjaRule()]
)
def test_rules_validate_state_shape(rule):
    with pytest.raises(ValueError, match="memory must have shape"):
        rule(torch.ones(2, 3), torch.ones(2, 4), torch.zeros(1, 4, 3))
