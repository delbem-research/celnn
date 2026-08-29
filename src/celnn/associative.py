"""Reusable key--value associative memories with local fast-weight updates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    import torch
except ImportError as exc:  # pragma: no cover - optional dependency branch
    raise ImportError(
        "Associative memory requires PyTorch. Install it with "
        "`pip install celnn[torch]`."
    ) from exc


@dataclass(frozen=True)
class AssociativeMemoryState:
    """Caller-owned key--value fast weights for independent sequences."""

    memory: torch.Tensor
    updates: int = 0

    def __post_init__(self) -> None:
        if self.memory.ndim != 3:
            raise ValueError("memory must have shape (batch, value, key).")
        if self.updates < 0:
            raise ValueError("updates must be non-negative.")

    @classmethod
    def zeros(
        cls,
        batch_size: int,
        value_size: int,
        key_size: int,
        *,
        like: torch.Tensor,
    ) -> "AssociativeMemoryState":
        dimensions = (batch_size, value_size, key_size)
        if any(
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < 1
            for value in dimensions
        ):
            raise ValueError("state dimensions must be positive integers.")
        return cls(like.new_zeros(dimensions))

    def reset(self) -> "AssociativeMemoryState":
        return AssociativeMemoryState(torch.zeros_like(self.memory))

    def detach(self) -> "AssociativeMemoryState":
        return AssociativeMemoryState(self.memory.detach(), self.updates)

    def to(self, *args: Any, **kwargs: Any) -> "AssociativeMemoryState":
        return AssociativeMemoryState(
            self.memory.to(*args, **kwargs), self.updates
        )


class DeltaHebbianRule:
    """Correct a key--value association using a local delta-Hebb update.

    For normalized key ``k``, target value ``v``, and fast matrix ``M``::

        prediction = M k
        M_next = retention * M + rate * (v - prediction) k^T

    ``rate`` and ``retention`` may be scalars or per-sample tensors supplied by
    a controller. The rule owns no hidden state and no trainable parameters.
    """

    def __init__(
        self,
        learning_rate: float = 0.1,
        retention: float = 0.99,
        *,
        normalize_keys: bool = True,
        epsilon: float = 1e-6,
    ) -> None:
        if learning_rate < 0:
            raise ValueError("learning_rate must be non-negative.")
        if not 0 <= retention <= 1:
            raise ValueError("retention must be between zero and one.")
        if epsilon <= 0:
            raise ValueError("epsilon must be positive.")
        self.learning_rate = float(learning_rate)
        self.retention = float(retention)
        self.normalize_keys = bool(normalize_keys)
        self.epsilon = float(epsilon)

    def normalize(self, vector: torch.Tensor) -> torch.Tensor:
        if not self.normalize_keys:
            return vector
        return torch.nn.functional.normalize(
            vector, dim=-1, eps=self.epsilon
        )

    @staticmethod
    def _coefficient(
        value: float | torch.Tensor,
        reference: torch.Tensor,
        name: str,
    ) -> torch.Tensor:
        coefficient = torch.as_tensor(
            value, dtype=reference.dtype, device=reference.device
        )
        if coefficient.ndim > 1 or (
            coefficient.ndim == 1
            and coefficient.shape[0] != reference.shape[0]
        ):
            raise ValueError(f"{name} must be scalar or have shape (batch,).")
        if coefficient.ndim == 1:
            coefficient = coefficient[:, None, None]
        return coefficient

    def __call__(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        memory: torch.Tensor,
        *,
        learning_rate: float | torch.Tensor | None = None,
        retention: float | torch.Tensor | None = None,
    ) -> torch.Tensor:
        if key.ndim != 2 or value.ndim != 2:
            raise ValueError(
                "key and value must have shape (batch, features)."
            )
        expected = (key.shape[0], value.shape[1], key.shape[1])
        if tuple(memory.shape) != expected:
            raise ValueError(f"memory must have shape {expected}.")
        if key.device != value.device or key.device != memory.device:
            raise ValueError(
                "key, value, and memory must use the same device."
            )

        key = self.normalize(key)
        prediction = torch.einsum("bvk,bk->bv", memory, key)
        error = value - prediction
        correction = torch.einsum("bv,bk->bvk", error, key)
        rate = self._coefficient(
            self.learning_rate if learning_rate is None else learning_rate,
            memory,
            "learning_rate",
        )
        keep = self._coefficient(
            self.retention if retention is None else retention,
            memory,
            "retention",
        )
        return keep * memory + rate * correction


class DeltaHebbianMemory(torch.nn.Module):
    """Functional memory composed from delta-Hebb read/write operations."""

    def __init__(
        self,
        key_size: int,
        value_size: int,
        rule: DeltaHebbianRule | None = None,
        *,
        detach_updates: bool = False,
        memory_limit: float | None = None,
    ) -> None:
        super().__init__()
        if key_size < 1 or value_size < 1:
            raise ValueError("key_size and value_size must be positive.")
        if memory_limit is not None and memory_limit <= 0:
            raise ValueError("memory_limit must be positive or None.")
        self.key_size = int(key_size)
        self.value_size = int(value_size)
        self.rule = rule or DeltaHebbianRule()
        self.detach_updates = bool(detach_updates)
        self.memory_limit = memory_limit

    def new_state(
        self, batch_size: int, *, like: torch.Tensor
    ) -> AssociativeMemoryState:
        return AssociativeMemoryState.zeros(
            batch_size,
            self.value_size,
            self.key_size,
            like=like,
        )

    def read(
        self, state: AssociativeMemoryState, query: torch.Tensor
    ) -> torch.Tensor:
        """Retrieve values for one or more queries without changing memory."""
        if query.ndim < 2 or query.shape[0] != state.memory.shape[0]:
            raise ValueError("query must have a matching batch axis.")
        if query.shape[-1] != self.key_size:
            raise ValueError(
                f"query last axis must have size {self.key_size}."
            )
        query = self.rule.normalize(query)
        return torch.einsum("b...k,bvk->b...v", query, state.memory)

    def write(
        self,
        state: AssociativeMemoryState,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        learning_rate: float | torch.Tensor | None = None,
        retention: float | torch.Tensor | None = None,
    ) -> AssociativeMemoryState:
        """Return a new state after writing one association per batch item."""
        memory = self.rule(
            key,
            value,
            state.memory,
            learning_rate=learning_rate,
            retention=retention,
        )
        if self.memory_limit is not None:
            memory = memory.clamp(-self.memory_limit, self.memory_limit)
        if self.detach_updates:
            memory = memory.detach()
        return AssociativeMemoryState(memory, state.updates + 1)
