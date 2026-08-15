"""Normalized associative fields for local differentiable fast memory."""

from __future__ import annotations

from dataclasses import dataclass

try:
    import torch
    import torch.nn.functional as F
except ImportError as exc:  # pragma: no cover - optional dependency branch
    raise ImportError(
        "Associative fields require PyTorch. Install them with "
        "`pip install celnn[torch]`."
    ) from exc


@dataclass(frozen=True)
class AssociativeFieldState:
    """Caller-owned numerator and normalizer fields.

    ``memory[b, i]`` stores the local key--value outer products of cell
    ``i``. ``normalizer[b, i]`` stores the matching key features. Keeping the
    state explicit lets callers propagate both fields over any topology.
    """

    memory: torch.Tensor
    normalizer: torch.Tensor
    updates: int = 0

    def __post_init__(self) -> None:
        if self.memory.ndim != 4:
            raise ValueError(
                "memory must have shape (batch, cells, value, key)."
            )
        if self.normalizer.ndim != 3:
            raise ValueError(
                "normalizer must have shape (batch, cells, key)."
            )
        expected = (
            self.memory.shape[0],
            self.memory.shape[1],
            self.memory.shape[3],
        )
        if tuple(self.normalizer.shape) != expected:
            raise ValueError(f"normalizer must have shape {expected}.")
        if self.memory.device != self.normalizer.device:
            raise ValueError("memory and normalizer must use the same device.")
        if self.memory.dtype != self.normalizer.dtype:
            raise ValueError("memory and normalizer must use the same dtype.")
        if self.updates < 0:
            raise ValueError("updates must be non-negative.")

    @classmethod
    def zeros(
        cls,
        batch_size: int,
        cells: int,
        value_size: int,
        key_size: int,
        *,
        like: torch.Tensor,
    ) -> "AssociativeFieldState":
        dimensions = (batch_size, cells, value_size, key_size)
        if any(
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < 1
            for value in dimensions
        ):
            raise ValueError("state dimensions must be positive integers.")
        memory = like.new_zeros(dimensions)
        normalizer = like.new_zeros((batch_size, cells, key_size))
        return cls(memory, normalizer)

    def reset(self) -> "AssociativeFieldState":
        return AssociativeFieldState(
            torch.zeros_like(self.memory),
            torch.zeros_like(self.normalizer),
        )

    def detach(self) -> "AssociativeFieldState":
        return AssociativeFieldState(
            self.memory.detach(),
            self.normalizer.detach(),
            self.updates,
        )

    def to(self, *args, **kwargs) -> "AssociativeFieldState":
        return AssociativeFieldState(
            self.memory.to(*args, **kwargs),
            self.normalizer.to(*args, **kwargs),
            self.updates,
        )


class NormalizedDeltaHebbianField(torch.nn.Module):
    """Local normalized key--value memory with Delta-Hebbian writes.

    For every cell, the read and write operations are::

        phi(z) = elu(z) + 1
        prediction = M phi(q) / (s^T phi(q) + epsilon)
        target = prediction + rate (value - prediction)
        s_next = retention s + rate phi(k)
        M_next is the local outer-product correction whose normalized
        response to k equals target

    The module deliberately does not prescribe how ``M`` and ``s`` move
    between cells. A CelNN, graph, or grid can propagate the explicit state
    before calling :meth:`read` or :meth:`write`.
    """

    def __init__(
        self,
        key_size: int,
        value_size: int,
        *,
        learning_rate: float = 0.1,
        retention: float = 0.99,
        epsilon: float = 1e-6,
        detach_updates: bool = False,
        memory_limit: float | None = None,
    ) -> None:
        super().__init__()
        if key_size < 1 or value_size < 1:
            raise ValueError("key_size and value_size must be positive.")
        if learning_rate < 0:
            raise ValueError("learning_rate must be non-negative.")
        if not 0 <= retention <= 1:
            raise ValueError("retention must be between zero and one.")
        if epsilon <= 0:
            raise ValueError("epsilon must be positive.")
        if memory_limit is not None and memory_limit <= 0:
            raise ValueError("memory_limit must be positive or None.")
        self.key_size = int(key_size)
        self.value_size = int(value_size)
        self.learning_rate = float(learning_rate)
        self.retention = float(retention)
        self.epsilon = float(epsilon)
        self.detach_updates = bool(detach_updates)
        self.memory_limit = memory_limit

    @staticmethod
    def feature_map(vector: torch.Tensor) -> torch.Tensor:
        """Map keys and queries to strictly positive kernel features."""
        return F.elu(vector) + 1.0

    def new_state(
        self, batch_size: int, cells: int, *, like: torch.Tensor
    ) -> AssociativeFieldState:
        return AssociativeFieldState.zeros(
            batch_size,
            cells,
            self.value_size,
            self.key_size,
            like=like,
        )

    def _validate(
        self, state: AssociativeFieldState, vector: torch.Tensor, name: str
    ) -> None:
        expected = (
            state.memory.shape[0],
            state.memory.shape[1],
            self.key_size,
        )
        if tuple(vector.shape) != expected:
            raise ValueError(f"{name} must have shape {expected}.")
        memory_features = tuple(state.memory.shape[2:])
        if memory_features != (self.value_size, self.key_size):
            raise ValueError(
                "state memory value/key axes do not match this field."
            )
        if vector.device != state.memory.device:
            raise ValueError(f"{name} and state must use the same device.")

    @staticmethod
    def _coefficient(
        value: float | torch.Tensor,
        reference: torch.Tensor,
        name: str,
    ) -> torch.Tensor:
        coefficient = torch.as_tensor(
            value, dtype=reference.dtype, device=reference.device
        )
        expected = reference.shape[:2]
        if coefficient.ndim == 0:
            return coefficient
        if tuple(coefficient.shape) != tuple(expected):
            raise ValueError(
                f"{name} must be scalar or have shape {expected}."
            )
        return coefficient

    def read(
        self, state: AssociativeFieldState, query: torch.Tensor
    ) -> torch.Tensor:
        """Read one normalized value from every cell without mutation."""
        self._validate(state, query, "query")
        features = self.feature_map(query)
        numerator = torch.einsum(
            "bcvk,bck->bcv", state.memory, features
        )
        denominator = torch.einsum(
            "bck,bck->bc", state.normalizer, features
        )
        return numerator / (denominator.unsqueeze(-1) + self.epsilon)

    def write(
        self,
        state: AssociativeFieldState,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        learning_rate: float | torch.Tensor | None = None,
        retention: float | torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> AssociativeFieldState:
        """Write one local association per cell and return a new state."""
        self._validate(state, key, "key")
        expected_value = (
            state.memory.shape[0],
            state.memory.shape[1],
            self.value_size,
        )
        if tuple(value.shape) != expected_value:
            raise ValueError(f"value must have shape {expected_value}.")
        if value.device != state.memory.device:
            raise ValueError("value and state must use the same device.")
        if mask is not None and tuple(mask.shape) != key.shape[:2]:
            raise ValueError("mask must match the batch and cell axes.")

        rate = self._coefficient(
            self.learning_rate if learning_rate is None else learning_rate,
            state.memory,
            "learning_rate",
        )
        keep = self._coefficient(
            self.retention if retention is None else retention,
            state.memory,
            "retention",
        )
        if mask is not None:
            active = mask.to(dtype=state.memory.dtype)
            rate = rate * active
            keep = keep * active + (1.0 - active)

        features = self.feature_map(key)
        prediction = self.read(state, key)
        target = prediction + rate[..., None] * (value - prediction)
        memory = keep[..., None, None] * state.memory
        normalizer = keep[..., None] * state.normalizer
        normalizer = normalizer + rate[..., None] * features
        target_denominator = torch.einsum(
            "bck,bck->bc", normalizer, features
        ) + self.epsilon
        current_numerator = torch.einsum(
            "bcvk,bck->bcv", memory, features
        )
        residual = target * target_denominator[..., None]
        residual = residual - current_numerator
        feature_energy = features.square().sum(dim=-1) + self.epsilon
        correction = torch.einsum(
            "bcv,bck->bcvk", residual / feature_energy[..., None], features
        )
        memory = memory + correction
        if self.memory_limit is not None:
            memory = memory.clamp(-self.memory_limit, self.memory_limit)
        if self.detach_updates:
            memory = memory.detach()
            normalizer = normalizer.detach()
        return AssociativeFieldState(memory, normalizer, state.updates + 1)
