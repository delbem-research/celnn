"""Template dataclass for reusable cellular operators."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import DTypeLike

from .exceptions import TemplateValidationError
from .validation import coerce_ndarray, validate_template_shapes


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
        """Create a template from a serialized dictionary."""
        template = cls(
            name=data["name"],
            feedback=data["feedback"],
            control=data["control"],
            bias=data.get("bias", 0.0),
            initial_state=data.get("initial_state"),
            description=data.get("description", ""),
            tags=list(data.get("tags", [])),
            metadata=deepcopy(data.get("metadata", {})),
        )
        return template.validate()
