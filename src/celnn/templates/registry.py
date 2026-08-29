"""Template registry for reusable named templates."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..core.exceptions import CelNNError
from ..core.templates import Template


@dataclass(slots=True)
class TemplateRegistry:
    """Registry for named templates."""

    _templates: dict[str, Template] = field(default_factory=dict)

    def register(self, template: Template, overwrite: bool = False) -> None:
        """Register a template by name."""
        template.validate()
        if template.name in self._templates and not overwrite:
            raise CelNNError(
                f"Template '{template.name}' is already registered."
            )
        self._templates[template.name] = template.copy()

    def get(self, name: str) -> Template:
        """Return a copy of a registered template."""
        try:
            return self._templates[name].copy()
        except KeyError as exc:
            raise CelNNError(f"Template '{name}' is not registered.") from exc

    def list(self) -> list[Template]:
        """Return registered templates as copies."""
        return [template.copy() for template in self._templates.values()]

    def names(self) -> list[str]:
        """Return registered template names."""
        return sorted(self._templates)

    def remove(self, name: str) -> Template:
        """Remove and return a registered template."""
        try:
            return self._templates.pop(name)
        except KeyError as exc:
            raise CelNNError(f"Template '{name}' is not registered.") from exc

    def to_dict(self) -> dict[str, Any]:
        """Serialize the registry to a JSON-friendly dictionary."""
        return {"templates": [template.to_dict() for template in self.list()]}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TemplateRegistry":
        """Restore a validated registry from a dictionary."""
        if not isinstance(data, dict):
            raise CelNNError("TemplateRegistry data must be a dictionary.")
        unknown = set(data) - {"templates"}
        if unknown:
            names = ", ".join(sorted(unknown))
            raise CelNNError(f"Unknown TemplateRegistry fields: {names}.")
        items = data.get("templates", [])
        if not isinstance(items, list):
            raise CelNNError("templates must be a list.")

        registry = cls()
        for item in items:
            if not isinstance(item, dict):
                raise CelNNError(
                    "Each registry template must be a dictionary."
                )
            registry.register(Template.from_dict(item))
        return registry


def builtin_templates() -> TemplateRegistry:
    """Return a registry with built-in templates preloaded."""
    from .diffusion import ONE_D_DIFFUSION, TWO_D_LAPLACIAN_DEMO
    from .image_processing import (
        CORNER_DETECTION,
        DIAGONAL_LINE_DETECTION,
        EDGE_DETECTION,
        INVERSION,
        SHARPENING_DEMO,
        SMOOTHING_DEMO,
    )
    from .logic import NOT_DEMO, THRESHOLD_DEMO
    from .pattern import DIFFUSION_LIKE, LOCAL_EXCITATION_GLOBAL_DAMPING_DEMO

    registry = TemplateRegistry()
    for template in (
        EDGE_DETECTION,
        INVERSION,
        CORNER_DETECTION,
        DIAGONAL_LINE_DETECTION,
        SMOOTHING_DEMO,
        SHARPENING_DEMO,
        NOT_DEMO,
        THRESHOLD_DEMO,
        DIFFUSION_LIKE,
        LOCAL_EXCITATION_GLOBAL_DAMPING_DEMO,
        ONE_D_DIFFUSION,
        TWO_D_LAPLACIAN_DEMO,
    ):
        registry.register(template)
    return registry
