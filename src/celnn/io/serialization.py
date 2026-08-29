"""Versioned JSON serialization helpers for celnn objects."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from ..core.exceptions import CelNNError
from ..core.network import CellularNetwork
from ..core.simulation import SimulationConfig
from ..core.templates import Template
from ..templates.registry import TemplateRegistry

_SCHEMA_VERSION = 1
_SUPPORTED_KINDS = {
    "network",
    "simulation_config",
    "template",
    "template_registry",
}


def save_json(data: dict[str, Any], path: str | Path) -> Path:
    """Atomically save a JSON-serializable dictionary to disk."""
    target = Path(path)
    serialized = json.dumps(data, indent=2, sort_keys=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
            temp_path = Path(handle.name)
        os.replace(temp_path, target)
    except Exception:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        raise
    return target


def load_json(path: str | Path) -> dict[str, Any]:
    """Load a JSON object from disk."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise CelNNError("Serialized CELNN JSON must contain an object.")
    return data


def _envelope(kind: str, data: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": _SCHEMA_VERSION,
        "kind": kind,
        "data": data,
    }


def _legacy_payload(kind: str, data: dict[str, Any]) -> dict[str, Any]:
    payload = dict(data)
    if kind == "network":
        payload.pop("state_shape", None)
        payload.pop("device", None)
        payload.pop("backend", None)
    elif kind == "simulation_config":
        payload.pop("stability_checks", None)
    return payload


def _load_artifact(path: str | Path, expected_kind: str) -> dict[str, Any]:
    raw = load_json(path)
    if "schema_version" not in raw:
        return _legacy_payload(expected_kind, raw)

    unknown = set(raw) - {"schema_version", "kind", "data"}
    if unknown:
        names = ", ".join(sorted(unknown))
        raise CelNNError(f"Unknown serialized artifact fields: {names}.")

    version = raw.get("schema_version")
    if type(version) is not int or version != _SCHEMA_VERSION:
        raise CelNNError(
            "Unsupported CELNN schema_version "
            f"{version!r}; expected {_SCHEMA_VERSION}."
        )

    kind = raw.get("kind")
    if not isinstance(kind, str) or kind not in _SUPPORTED_KINDS:
        raise CelNNError(f"Unsupported CELNN artifact kind {kind!r}.")
    if kind != expected_kind:
        raise CelNNError(
            f"Expected CELNN artifact kind {expected_kind!r}, got {kind!r}."
        )

    payload = raw.get("data")
    if not isinstance(payload, dict):
        raise CelNNError("Serialized CELNN artifact data must be an object.")
    return payload


def save_template_json(template: Template, path: str | Path) -> Path:
    """Serialize a template to schema-v1 JSON."""
    return save_json(_envelope("template", template.to_dict()), path)


def load_template_json(path: str | Path) -> Template:
    """Load a schema-v1 or legacy-v0 template JSON artifact."""
    return Template.from_dict(_load_artifact(path, "template"))


def save_config_json(config: SimulationConfig, path: str | Path) -> Path:
    """Serialize a simulation configuration to schema-v1 JSON."""
    return save_json(
        _envelope("simulation_config", config.to_dict()),
        path,
    )


def load_config_json(path: str | Path) -> SimulationConfig:
    """Load a schema-v1 or legacy-v0 simulation configuration."""
    return SimulationConfig.from_dict(
        _load_artifact(path, "simulation_config")
    )


def save_registry_json(registry: TemplateRegistry, path: str | Path) -> Path:
    """Serialize a template registry to schema-v1 JSON."""
    return save_json(
        _envelope("template_registry", registry.to_dict()),
        path,
    )


def load_registry_json(path: str | Path) -> TemplateRegistry:
    """Load a schema-v1 or legacy-v0 template registry."""
    return TemplateRegistry.from_dict(
        _load_artifact(path, "template_registry")
    )


def save_network_json(network: CellularNetwork, path: str | Path) -> Path:
    """Serialize a network to schema-v1 JSON."""
    return save_json(_envelope("network", network.to_dict()), path)


def load_network_json(
    path: str | Path,
    *,
    device: str = "cpu",
) -> CellularNetwork:
    """Load a schema-v1 or legacy-v0 network on the requested device."""
    return CellularNetwork.from_dict(
        _load_artifact(path, "network"),
        device=device,
    )
