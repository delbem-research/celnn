import json

import numpy as np
import pytest

from celnn import CellularNetwork, SimulationConfig
from celnn.core.exceptions import CelNNError, SolverError
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
    assert raw["schema_version"] == 1
    assert raw["kind"] == "template"
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
    assert raw["schema_version"] == 1
    assert raw["kind"] == "network"
    assert "device" not in raw["data"]
    assert "backend" not in raw["data"]
    restored = load_network_json(path)
    assert restored.to_dict() == net.to_dict()
    assert restored.device == "cpu"

    config = SimulationConfig(t_end=2.0, dt=0.1, return_trajectory=True)
    config_path = tmp_path / "config.json"
    save_config_json(config, config_path)
    assert load_config_json(config_path) == config


def test_legacy_v0_network_is_read_but_not_rewritten_as_v0(tmp_path):
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

    save_network_json(restored, path)
    rewritten = json.loads(path.read_text())
    assert rewritten["schema_version"] == 1
    assert rewritten["kind"] == "network"


def test_legacy_v0_config_drops_removed_stability_field(tmp_path):
    legacy = {
        "t_start": 0.0,
        "t_end": 1.0,
        "dt": 0.1,
        "solver": "euler",
        "return_trajectory": False,
        "store_every": 1,
        "stability_checks": True,
        "progress": False,
    }
    path = tmp_path / "legacy-config.json"
    path.write_text(json.dumps(legacy), encoding="utf-8")
    restored = load_config_json(path)
    assert restored == SimulationConfig(dt=0.1)


def test_schema_version_and_kind_are_validated(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(
        json.dumps({"schema_version": 99, "kind": "network", "data": {}}),
        encoding="utf-8",
    )
    with pytest.raises(CelNNError, match="schema_version"):
        load_network_json(path)

    path.write_text(
        json.dumps(
            {"schema_version": 1, "kind": "template", "data": {}}
        ),
        encoding="utf-8",
    )
    with pytest.raises(CelNNError, match="Expected CELNN artifact kind"):
        load_network_json(path)


def test_public_deserialization_rejects_truthy_string_booleans():
    with pytest.raises(SolverError, match="boolean"):
        SimulationConfig.from_dict({"return_trajectory": "false"})


def test_v1_network_unknown_fields_are_rejected(tmp_path):
    net = CellularNetwork(input=np.ones(3))
    payload = net.to_dict()
    payload["typo"] = 1
    path = tmp_path / "bad-network.json"
    path.write_text(
        json.dumps(
            {"schema_version": 1, "kind": "network", "data": payload}
        ),
        encoding="utf-8",
    )
    with pytest.raises(CelNNError, match="Unknown CellularNetwork fields"):
        load_network_json(path)
