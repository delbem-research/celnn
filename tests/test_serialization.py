import json
from types import SimpleNamespace

import numpy as np
import pytest

from celnn import CellularNetwork, SimulationConfig
from celnn.core.exceptions import ShapeMismatchError
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


def test_legacy_execution_device_is_preserved_and_not_rewritten(
    tmp_path, monkeypatch
):
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

    def fake_get_backend(device):
        name = "cupy" if device == "gpu" else "numpy"
        return SimpleNamespace(name=name)

    monkeypatch.setattr("celnn.core.network.get_backend", fake_get_backend)

    restored = load_network_json(path)
    assert restored.state.shape == signal.shape
    assert restored.device == "gpu"
    assert restored.backend.name == "cupy"

    overridden = load_network_json(path, device="cpu")
    assert overridden.device == "cpu"
    assert overridden.backend.name == "numpy"

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


def test_network_loader_rejects_incompatible_current_state_shape(tmp_path):
    payload = CellularNetwork(input=np.ones(3)).to_dict()
    payload["current_state"] = [1.0]
    path = tmp_path / "invalid-network.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ShapeMismatchError, match="shape mismatch"):
        load_network_json(path)
