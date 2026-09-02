"""Composable fast-weight plasticity for PyTorch models.

The primitives in this module are deliberately independent from CelNN grids.
Rules compute local updates, :class:`PlasticityState` owns transient memory,
and :class:`Plasticity` composes that memory with any slow weight tensor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

try:
    import torch
except ImportError as _exc:  # pragma: no cover - optional dependency branch
    raise ImportError(
        "Plasticity requires PyTorch. Install it with "
        "`pip install celnn[torch]`."
    ) from _exc


@dataclass(frozen=True)
class PlasticityState:
    """Immutable, transient fast-weight memory for a batch of sequences."""

    memory: torch.Tensor
    updates: int = 0

    def __post_init__(self) -> None:
        if self.memory.ndim != 3:
            raise ValueError("memory must have shape (batch, output, input).")
        if self.updates < 0:
            raise ValueError("updates must be non-negative.")

    @classmethod
    def zeros(
        cls,
        batch_size: int,
        output_features: int,
        input_features: int,
        *,
        like: torch.Tensor,
    ) -> "PlasticityState":
        """Create empty memory matching a reference tensor's device/dtype."""
        dimensions = (batch_size, output_features, input_features)
        if any(
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < 1
            for value in dimensions
        ):
            raise ValueError("state dimensions must be positive integers.")
        return cls(like.new_zeros(dimensions))

    def reset(self) -> "PlasticityState":
        """Return a zeroed state with the same shape, device, and dtype."""
        return PlasticityState(torch.zeros_like(self.memory))

    def detach(self) -> "PlasticityState":
        """Cut history while retaining the current fast weights."""
        return PlasticityState(self.memory.detach(), self.updates)

    def to(self, *args: Any, **kwargs: Any) -> "PlasticityState":
        """Move or cast the memory like :meth:`torch.Tensor.to`."""
        return PlasticityState(self.memory.to(*args, **kwargs), self.updates)


@runtime_checkable
class PlasticityRule(Protocol):
    """Contract for a local rule that returns the next fast-weight tensor."""

    def __call__(
        self,
        pre: torch.Tensor,
        post: torch.Tensor,
        memory: torch.Tensor,
    ) -> torch.Tensor:
        """Update ``memory`` from paired pre/post-synaptic activities."""
        ...


def _activities(
    pre: torch.Tensor,
    post: torch.Tensor,
    memory: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if pre.ndim < 2 or post.ndim < 2:
        raise ValueError("activities must have a batch and feature axis.")
    if pre.shape[:-1] != post.shape[:-1]:
        raise ValueError("pre and post activities must share sample axes.")
    expected = (pre.shape[0], post.shape[-1], pre.shape[-1])
    if tuple(memory.shape) != expected:
        raise ValueError(f"memory must have shape {expected}.")
    if pre.device != post.device or pre.device != memory.device:
        raise ValueError("activities and memory must use the same device.")
    if pre.ndim == 2:
        return pre.unsqueeze(1), post.unsqueeze(1)
    return pre.flatten(1, -2), post.flatten(1, -2)


def _mean_outer(pre: torch.Tensor, post: torch.Tensor) -> torch.Tensor:
    return torch.einsum("bso,bsi->boi", post, pre) / pre.shape[1]


class HebbianRule:
    """Hebbian correlation with optional decay: ``λH + η E[y xᵀ]``."""

    learning_rate: float
    decay: float

    def __init__(
        self, learning_rate: float = 0.01, decay: float = 1.0
    ) -> None:
        if learning_rate < 0:
            raise ValueError("learning_rate must be non-negative.")
        if not 0 <= decay <= 1:
            raise ValueError("decay must be between zero and one.")
        self.learning_rate = float(learning_rate)
        self.decay = float(decay)

    def __call__(
        self,
        pre: torch.Tensor,
        post: torch.Tensor,
        memory: torch.Tensor,
    ) -> torch.Tensor:
        pre_samples, post_samples = _activities(pre, post, memory)
        correlation = _mean_outer(pre_samples, post_samples)
        return self.decay * memory + self.learning_rate * correlation


class OjaRule:
    """Normalized Hebbian learning using Oja's local stabilizing term."""

    learning_rate: float
    decay: float

    def __init__(
        self, learning_rate: float = 0.01, decay: float = 1.0
    ) -> None:
        if learning_rate < 0:
            raise ValueError("learning_rate must be non-negative.")
        if not 0 <= decay <= 1:
            raise ValueError("decay must be between zero and one.")
        self.learning_rate = float(learning_rate)
        self.decay = float(decay)

    def __call__(
        self,
        pre: torch.Tensor,
        post: torch.Tensor,
        memory: torch.Tensor,
    ) -> torch.Tensor:
        pre_samples, post_samples = _activities(pre, post, memory)
        correlation = _mean_outer(pre_samples, post_samples)
        post_energy = post_samples.square().mean(dim=1).unsqueeze(-1)
        update = correlation - post_energy * memory
        return self.decay * memory + self.learning_rate * update


class Plasticity(torch.nn.Module):
    """Compose slow weights with rule-driven transient fast weights."""

    alpha: torch.Tensor
    rule: PlasticityRule
    detach_updates: bool
    memory_limit: float | None

    def __init__(
        self,
        rule: PlasticityRule,
        alpha: float = 1.0,
        *,
        learnable_alpha: bool = False,
        detach_updates: bool = True,
        memory_limit: float | None = None,
    ) -> None:
        super().__init__()
        if memory_limit is not None and memory_limit <= 0:
            raise ValueError("memory_limit must be positive or None.")
        alpha_tensor = torch.tensor(float(alpha))
        if learnable_alpha:
            self.alpha = torch.nn.Parameter(alpha_tensor)
        else:
            self.register_buffer("alpha", alpha_tensor)
        self.rule = rule
        self.detach_updates = bool(detach_updates)
        self.memory_limit = memory_limit

    def new_state(
        self, batch_size: int, weight: torch.Tensor
    ) -> PlasticityState:
        """Create memory compatible with an ``(output, input)`` weight."""
        if weight.ndim != 2:
            raise ValueError("weight must have shape (output, input).")
        return PlasticityState.zeros(
            batch_size, weight.shape[0], weight.shape[1], like=weight
        )

    def effective_weight(
        self, weight: torch.Tensor, state: PlasticityState
    ) -> torch.Tensor:
        """Return per-sample ``slow + alpha * fast`` weights."""
        if weight.ndim != 2:
            raise ValueError("weight must have shape (output, input).")
        if tuple(state.memory.shape[1:]) != tuple(weight.shape):
            raise ValueError("state memory is incompatible with weight.")
        return weight.unsqueeze(0) + self.alpha * state.memory

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
        """Return output and the next caller-owned plasticity state."""
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
