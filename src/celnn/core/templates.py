"""Template dataclass for reusable cellular operators."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import DTypeLike

from .exceptions import TemplateValidationError
from .validation import coerce_ndarray, validate_template_shapes

_ALLOWED_KEYS = {
    "name",
    "feedback",
    "control",
    "bias",
    "initial_state",
    "description",
    "tags",
    "metadata",
}


@dataclass(slots=True)
class Template:
    """Reusable feedback/control template bundle."""

    name: str
    feedback: Any
    control: Any
    bias: Any = 0.0
    initial_state: Any | None = None
    description: str = ""
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> "Template":
        """Validate template consistency."""
        if not isinstance(self.name, str) or not self.name:
            raise TemplateValidationError(
                "Template name must be a non-empty string."
            )
        if not isinstance(self.description, str):
            raise TemplateValidationError(
                "Template description must be a string."
            )
        if not isinstance(self.tags, list) or not all(
            isinstance(tag, str) for tag in self.tags
        ):
            raise TemplateValidationError(
                "Template tags must be a list of strings."
            )
        if not isinstance(self.metadata, dict):
            raise TemplateValidationError(
                "Template metadata must be a dictionary."
            )

        feedback = coerce_ndarray(self.feedback, dtype=float, name="feedback")
        control = coerce_ndarray(self.control, dtype=float, name="control")
        validate_template_shapes(feedback, control, feedback.ndim)
        if self.initial_state is not None:
            initial_state = coerce_ndarray(
                self.initial_state, dtype=float, name="initial_state"
            )
            if initial_state.ndim != feedback.ndim:
                raise TemplateValidationError(
                    "Template initial_state must have the same "
                    "dimensionality as the template."
                )
        return self

    def as_arrays(
        self, *, dtype: DTypeLike | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """Return feedback, control, and optional initial state as arrays."""
        self.validate()
        feedback = np.asarray(self.feedback, dtype=dtype or float)
        control = np.asarray(self.control, dtype=dtype or float)
        initial_state = None
        if self.initial_state is not None:
            initial_state = np.asarray(
                self.initial_state, dtype=dtype or float
            )
        return feedback, control, initial_state

    def copy(self) -> "Template":
        """Return a deep copy of the template."""
        return deepcopy(self)

    def with_bias(self, bias: Any) -> "Template":
        """Return a copied template with a different bias."""
        new_template = self.copy()
        new_template.bias = bias
        return new_template

    def to_dict(self) -> dict[str, Any]:
        """Serialize the template to a JSON-friendly dictionary."""
        self.validate()
        feedback, control, initial_state = self.as_arrays()
        return {
            "name": self.name,
            "feedback": feedback.tolist(),
            "control": control.tolist(),
            "bias": np.asarray(self.bias).tolist(),
            "initial_state": None
            if initial_state is None
            else initial_state.tolist(),
            "description": self.description,
            "tags": list(self.tags),
            "metadata": deepcopy(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Template":
        """Create a validated template from a dictionary."""
        if not isinstance(data, dict):
            raise TemplateValidationError(
                "Template data must be a dictionary."
            )
        unknown = set(data) - _ALLOWED_KEYS
        if unknown:
            names = ", ".join(sorted(unknown))
            raise TemplateValidationError(f"Unknown Template fields: {names}.")
        for required in ("name", "feedback", "control"):
            if required not in data:
                raise TemplateValidationError(
                    f"Template data is missing {required!r}."
                )
        description = data.get("description", "")
        tags = data.get("tags", [])
        metadata = data.get("metadata", {})
        if not isinstance(description, str):
            raise TemplateValidationError("description must be a string.")
        if not isinstance(tags, list) or not all(
            isinstance(tag, str) for tag in tags
        ):
            raise TemplateValidationError("tags must be a list of strings.")
        if not isinstance(metadata, dict):
            raise TemplateValidationError("metadata must be a dictionary.")

        template = cls(
            name=data["name"],
            feedback=data["feedback"],
            control=data["control"],
            bias=data.get("bias", 0.0),
            initial_state=data.get("initial_state"),
            description=description,
            tags=list(tags),
            metadata=deepcopy(metadata),
        )
        return template.validate()
