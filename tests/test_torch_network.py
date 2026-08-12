"""The differentiable network is the same CelNN, with learnable templates."""

from __future__ import annotations

import numpy as np
import pytest

import celnn
from celnn import CellularNetwork, SimulationConfig
from celnn.core.templates import Template

torch = pytest.importorskip("torch", reason="PyTorch is optional")
DifferentiableCellularNetwork = celnn.DifferentiableCellularNetwork

FEEDBACK = [0.2, -0.3, 1.0, -0.3, 0.2]
CONTROL = [0.1, 0.2, 0.6, 0.2, 0.1]
BIAS = 0.05
DT = 0.1
STEPS = 15


def test_differentiable_network_is_part_of_the_top_level_api():
    from celnn import DifferentiableCellularNetwork as PublicNetwork

    assert PublicNetwork is DifferentiableCellularNetwork


def _template(name: str = "probe") -> Template:
    return Template(
        name=name, feedback=FEEDBACK, control=CONTROL, bias=BIAS
    )


def _reference(u: np.ndarray, *, boundary: str = "constant") -> np.ndarray:
    net = CellularNetwork(
        input=u,
        feedback=np.array(FEEDBACK),
        control=np.array(CONTROL),
        bias=BIAS,
        activation="piecewise_linear",
        boundary=boundary,
    )
    return net.run(
        SimulationConfig(t_end=DT * STEPS, dt=DT, solver="euler")
    ).state


@pytest.mark.parametrize(
    "boundary", ["constant", "wrap", "reflect", "nearest", "mirror"]
)
def test_matches_the_reference_simulator(boundary):
    """Same templates, same trajectory -- no new architecture was invented."""
    rng = np.random.default_rng(0)
    u = rng.normal(size=24)

    net = DifferentiableCellularNetwork.from_template(
        _template(),
        radius=2,
        boundary=boundary,
        dt=DT,
        steps=STEPS,
    ).double()
    got = net(torch.tensor(u, dtype=torch.float64))

    np.testing.assert_allclose(
        got.detach().numpy(),
        _reference(u, boundary=boundary),
        rtol=1e-9,
        atol=1e-9,
    )


def test_templates_are_learnable_parameters():
    net = DifferentiableCellularNetwork.from_template(_template(), radius=2)
    names = {name for name, _ in net.named_parameters()}
    assert names == {"feedback", "control", "bias"}
    assert all(p.requires_grad for p in net.parameters())


def test_frozen_construction_yields_buffers_not_parameters():
    net = DifferentiableCellularNetwork.from_template(
        _template(), radius=2, trainable=False
    )
    assert list(net.parameters()) == []


def test_round_trip_through_template_returns_to_the_simulator():
    """Learned templates must be runnable by the reference implementation."""
    rng = np.random.default_rng(1)
    u = rng.normal(size=20)

    net = DifferentiableCellularNetwork.from_template(
        _template(), radius=2, dt=DT, steps=STEPS
    ).double()
    recovered = net.to_template(name="recovered")

    assert isinstance(recovered, Template)
    np.testing.assert_allclose(
        np.asarray(recovered.feedback, dtype=float), FEEDBACK
    )

    reference = CellularNetwork.from_template(recovered, input=u)
    result = reference.run(
        SimulationConfig(t_end=DT * STEPS, dt=DT, solver="euler")
    )
    np.testing.assert_allclose(
        result.state,
        net(torch.tensor(u, dtype=torch.float64)).detach().numpy(),
        rtol=1e-9,
        atol=1e-9,
    )


def test_multichannel_reduces_to_the_classical_case_at_one_channel():
    rng = np.random.default_rng(2)
    u = rng.normal(size=18)

    scalar = DifferentiableCellularNetwork.from_template(
        _template(), radius=2, dt=DT, steps=STEPS
    ).double()
    channelled = DifferentiableCellularNetwork.from_template(
        _template(), radius=2, channels=1, dt=DT, steps=STEPS
    ).double()

    flat = scalar(torch.tensor(u, dtype=torch.float64))
    shaped = channelled(
        torch.tensor(u, dtype=torch.float64).reshape(1, 18, 1)
    )

    np.testing.assert_allclose(
        shaped.reshape(-1).detach().numpy(),
        flat.detach().numpy(),
        rtol=1e-12,
        atol=1e-12,
    )


def test_multichannel_parameter_shapes_are_diagonal():
    net = DifferentiableCellularNetwork(radius=2, channels=16)
    assert net.feedback.shape == (5, 16)
    assert net.control.shape == (5, 16)
    assert net.bias.shape == (16,)
    total = sum(p.numel() for p in net.parameters())
    assert total == 5 * 16 + 5 * 16 + 16

    field = torch.randn(3, 30, 16)
    assert net(field).shape == (3, 30, 16)


def test_gradients_reach_every_template_through_all_steps():
    net = DifferentiableCellularNetwork(
        radius=2, channels=8, steps=16
    ).double()
    with torch.no_grad():
        net.feedback.normal_(std=0.1)
        net.control.normal_(std=0.1)

    net(torch.randn(4, 32, 8, dtype=torch.float64)).pow(2).mean().backward()

    for name, parameter in net.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name


def test_gradcheck_on_a_small_network():
    net = DifferentiableCellularNetwork(
        radius=1, channels=2, steps=3, activation="tanh_activation"
    ).double()
    with torch.no_grad():
        net.feedback.normal_(std=0.1)
        net.control.normal_(std=0.1)
    u = torch.randn(1, 6, 2, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(net, (u,), eps=1e-6, atol=1e-6)


def test_a_known_template_can_be_recovered_by_gradient_descent():
    torch.manual_seed(0)
    target = DifferentiableCellularNetwork(
        radius=1, channels=4, steps=8
    ).double()
    with torch.no_grad():
        target.feedback.normal_(std=0.2)
        target.control.normal_(std=0.2)

    u = torch.randn(16, 24, 4, dtype=torch.float64)
    with torch.no_grad():
        goal = target(u)

    learner = DifferentiableCellularNetwork(
        radius=1, channels=4, steps=8
    ).double()
    optimizer = torch.optim.Adam(learner.parameters(), lr=0.05)
    first = torch.nn.functional.mse_loss(learner(u), goal).item()
    for _ in range(300):
        optimizer.zero_grad()
        loss = torch.nn.functional.mse_loss(learner(u), goal)
        loss.backward()
        optimizer.step()

    assert loss.item() < first * 0.1, f"{first:.3e} -> {loss.item():.3e}"


def test_semi_implicit_method_is_available():
    net = DifferentiableCellularNetwork(
        radius=1, channels=2, steps=4, method="semi_implicit_euler"
    ).double()
    assert net(torch.randn(1, 8, 2, dtype=torch.float64)).shape == (1, 8, 2)


def test_unknown_method_is_rejected():
    with pytest.raises(ValueError, match="method"):
        DifferentiableCellularNetwork(radius=1, method="rk4")
