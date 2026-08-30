import json

import numpy as np

from celnn import CellularNetwork, SimulationConfig
from celnn.io.serialization import (
    load_config_json,
    load_network_json,
    load_registry_json,
    load_template_json,
    save_config_json,
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
    raw = json.loads(path.read_text())
    assert raw == template.to_dict()
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
    raw = json.loads(path.read_text())
    assert "schema_version" not in raw
    assert "input" in raw
    assert "device" not in raw
    assert "backend" not in raw
    restored = load_network_json(path)
    assert restored.to_dict() == net.to_dict()
    assert restored.device == "cpu"

    config = SimulationConfig(
        t_end=2.0,
        dt=0.1,
        return_trajectory=True,
        stability_checks=False,
    )
    config_path = tmp_path / "config.json"
    save_config_json(config, config_path)
    assert load_config_json(config_path) == config


def test_legacy_network_execution_fields_are_accepted_but_not_rewritten(tmp_path):
    signal = np.ones(5)
    net = CellularNetwork(
        input=signal,
        feedback=[0.0, 0.0, 0.0],
        control=[0.0, 1.0, 0.0],
        activation="identity",
        device="cpu",
    )
    legacy = net.to_dict()
    legacy["state_shape"] = [5]
    legacy["device"] = "gpu"
    legacy["backend"] = "cupy"
    path = tmp_path / "legacy-network.json"
    path.write_text(json.dumps(legacy), encoding="utf-8")

    restored = load_network_json(path)
    assert restored.state.shape == signal.shape
    assert restored.device == "cpu"
    assert restored.backend.name == "numpy"

    save_network_json(restored, path)
    rewritten = json.loads(path.read_text())
    assert "schema_version" not in rewritten
    assert "device" not in rewritten
    assert "backend" not in rewritten


def test_config_preserves_legacy_permissive_deserialization():
    restored = SimulationConfig.from_dict(
        {
            "return_trajectory": "false",
            "stability_checks": "false",
            "unknown_future_field": 1,
        }
    )
    assert restored.return_trajectory is True
    assert restored.stability_checks is True


def test_template_and_registry_ignore_unknown_fields():
    template = Template.from_dict(
        {
            "name": "demo",
            "feedback": [0.0, 1.0, 0.0],
            "control": [0.0, 0.0, 0.0],
            "unknown_future_field": 1,
        }
    )
    registry = TemplateRegistry.from_dict(
        {
            "templates": [template.to_dict()],
            "unknown_future_field": 1,
        }
    )
    assert registry.names() == ["demo"]
