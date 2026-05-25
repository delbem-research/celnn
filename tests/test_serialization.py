import numpy as np

from celnn import CellularNetwork, SimulationConfig
from celnn.io.serialization import (
    load_network_json,
    load_registry_json,
    load_template_json,
    save_network_json,
    save_registry_json,
    save_template_json,
)
from celnn.templates import Template, TemplateRegistry


def test_template_serialization_roundtrip(tmp_path):
    template = Template(
        name="demo",
        feedback=[[0.0, 1.0, 0.0], [1.0, 2.0, 1.0], [0.0, 1.0, 0.0]],
        control=[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
        bias=0.0,
    )
    path = tmp_path / "template.json"
    save_template_json(template, path)
    restored = load_template_json(path)
    assert restored.to_dict() == template.to_dict()


def test_registry_serialization_roundtrip(tmp_path):
    registry = TemplateRegistry()
    registry.register(
        Template(
            name="demo", feedback=[0.0, 1.0, 0.0], control=[0.0, 0.0, 0.0]
        )
    )
    path = tmp_path / "registry.json"
    save_registry_json(registry, path)
    restored = load_registry_json(path)
    assert restored.names() == ["demo"]


def test_network_and_config_serialization_roundtrip(tmp_path):
    signal = np.ones(5)
    net = CellularNetwork(
        input=signal,
        feedback=[0.0, 0.0, 0.0],
        control=[0.0, 1.0, 0.0],
        activation="identity",
    )
    path = tmp_path / "network.json"
    save_network_json(net, path)
    restored = load_network_json(path)
    assert restored.to_dict() == net.to_dict()

    config = SimulationConfig(t_end=2.0, dt=0.1, return_trajectory=True)
    assert SimulationConfig.from_dict(config.to_dict()) == config


def test_network_from_dict_accepts_legacy_state_shape_field():
    signal = np.ones(5)
    net = CellularNetwork(
        input=signal,
        feedback=[0.0, 0.0, 0.0],
        control=[0.0, 1.0, 0.0],
        activation="identity",
    )
    serialized = net.to_dict()
    serialized["state_shape"] = [5]
    restored = CellularNetwork.from_dict(serialized)
    assert restored.state.shape == signal.shape
