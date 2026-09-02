"""Differentiable plasticity utilities built on PyTorch."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from .utils.doc import optional_dependency_message

try:
    import torch
except ImportError as exc:  # pragma: no cover - depends on environment
    raise ImportError(optional_dependency_message("torch", "torch")) from exc


@dataclass(frozen=True)
class PlasticityState:
    """Caller-owned mutable-history state for plasticity operations."""

    memory: torch.Tensor
    updates: int = 0

    def reset(self) -> PlasticityState:
        """Return a zeroed state with the same shape, dtype, and device."""
        return PlasticityState(torch.zeros_like(self.memory), 0)

    def detach(self) -> PlasticityState:
        """Return a state detached from autograd history."""
        return PlasticityState(self.memory.detach(), self.updates)

    def to(self, *args: Any, **kwargs: Any) -> PlasticityState:
        """Move/cast the state using :meth:`torch.Tensor.to` semantics."""
        return PlasticityState(self.memory.to(*args, **kwargs), self.updates)


class PlasticityRule(ABC):
    """Base class for local plasticity update rules."""

    @abstractmethod
    def __call__(
        self,
        pre: torch.Tensor,
        post: torch.Tensor,
        memory: torch.Tensor,
    ) -> torch.Tensor:
        """Return the updated memory tensor."""
        raise NotImplementedError


@dataclass(frozen=True)
class HebbianRule(PlasticityRule):
    """Hebbian update using an exponentially decayed outer product."""

    learning_rate: float = 1e-2
    decay: float = 0.0

    def __post_init__(self) -> None:
        if self.learning_rate < 0.0:
            raise ValueError("learning_rate must be non-negative.")
        if not 0.0 <= self.decay <= 1.0:
            raise ValueError("decay must lie in [0, 1].")

    def __call__(
        self,
        pre: torch.Tensor,
        post: torch.Tensor,
        memory: torch.Tensor,
    ) -> torch.Tensor:
        delta = torch.einsum("b...o,b...i->boi", post, pre)
        delta = delta / max(pre[0].numel() // pre.shape[-1], 1)
        return (1.0 - self.decay) * memory + self.learning_rate * delta


@dataclass(frozen=True)
class OjaRule(PlasticityRule):
    """Oja-style normalized Hebbian update."""

    learning_rate: float = 1e-2
    decay: float = 0.0

    def __post_init__(self) -> None:
        if self.learning_rate < 0.0:
            raise ValueError("learning_rate must be non-negative.")
        if not 0.0 <= self.decay <= 1.0:
            raise ValueError("decay must lie in [0, 1].")

    def __call__(
        self,
        pre: torch.Tensor,
        post: torch.Tensor,
        memory: torch.Tensor,
    ) -> torch.Tensor:
        hebbian = torch.einsum("b...o,b...i->boi", post, pre)
        normalization = torch.einsum("b...o,boi->boi", post**2, memory)
        delta = hebbian - normalization
        delta = delta / max(pre[0].numel() // pre.shape[-1], 1)
        return (1.0 - self.decay) * memory + self.learning_rate * delta


@dataclass
class Plasticity:
    """Apply a local plasticity rule to caller-owned explicit state."""

    rule: PlasticityRule
    memory_limit: float | None = None
    detach_updates: bool = False

    def __post_init__(self) -> None:
        if self.memory_limit is not None and self.memory_limit <= 0.0:
            raise ValueError("memory_limit must be positive when provided.")

    def new_state(
        self,
        batch_size: int,
        base_weight: torch.Tensor,
    ) -> PlasticityState:
        """Create zeroed per-sample memory matching ``base_weight``."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        shape = (batch_size, *base_weight.shape)
        memory = torch.zeros(
            shape,
            dtype=base_weight.dtype,
            device=base_weight.device,
        )
        return PlasticityState(memory)

    @staticmethod
    def effective_weight(
        base_weight: torch.Tensor,
        state: PlasticityState,
    ) -> torch.Tensor:
        """Return base weight plus caller-owned fast-weight memory."""
        return base_weight.unsqueeze(0) + state.memory

    def update(
        self,
        state: PlasticityState,
        pre: torch.Tensor,
        post: torch.Tensor,
    ) -> PlasticityState:
        """Return the next state without mutating the caller's state."""
        memory = self.rule(pre, post, state.memory)
        if self.memory_limit is not None:
            memory = memory.clamp(-self.memory_limit, self.memory_limit)
        if self.detach_updates:
            memory = memory.detach()
        return PlasticityState(memory, state.updates + 1)


class PlasticLinear(torch.nn.Module):
    """Reusable linear layer with caller-owned, per-sample fast weights."""

    linear: torch.nn.Linear
    plasticity: Plasticity

    def __init__(
        self,
        input_features: int,
        output_features: int,
        plasticity: Plasticity,
        *,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(
            input_features, output_features, bias=bias
        )
        self.plasticity = plasticity

    def new_state(self, batch_size: int) -> PlasticityState:
        return self.plasticity.new_state(batch_size, self.linear.weight)

    def forward(
        self,
        input: torch.Tensor,
        state: PlasticityState,
        *,
        update: bool = True,
    ) -> tuple[torch.Tensor, PlasticityState]:
        """Return the layer output and the next caller-owned plasticity state."""
        if input.ndim < 2:
            raise ValueError("input must have a batch and feature axis.")
        if input.shape[0] != state.memory.shape[0]:
            raise ValueError("input and state batch sizes must match.")
        weight = self.plasticity.effective_weight(self.linear.weight, state)
        output = torch.einsum("b...i,boi->b...o", input, weight)
        if self.linear.bias is not None:
            output = output + self.linear.bias
        next_state = (
            self.plasticity.update(state, input, output) if update else state
        )
        return output, next_state


__all__ = [
    "HebbianRule",
    "OjaRule",
    "PlasticLinear",
    "Plasticity",
    "PlasticityRule",
    "PlasticityState",
]
