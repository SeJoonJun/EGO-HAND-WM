"""Typed canonical sample/batch contract and variable-time collation."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Iterable

import torch
from torch.nn.utils.rnn import pad_sequence

from ego_hand_wm.contracts.schema import GEOMETRY_DIM, STREAM_NAMES


@dataclass
class CanonicalBatch:
    history_time: torch.Tensor
    history_query_mask: torch.Tensor
    history_state: torch.Tensor
    history_stream_mask: torch.Tensor
    future_time: torch.Tensor
    future_query_stream_mask: torch.Tensor
    future_state: torch.Tensor
    future_stream_mask: torch.Tensor
    text: list[str]
    intrinsics: torch.Tensor
    context_images: torch.Tensor | None = None
    context_visual_features: torch.Tensor | None = None
    future_visual_latents: torch.Tensor | None = None
    history_fingertips: torch.Tensor | None = None
    future_fingertips: torch.Tensor | None = None
    metadata: list[dict[str, Any]] | None = None

    def to(self, device: torch.device | str, non_blocking: bool = False) -> "CanonicalBatch":
        values: dict[str, Any] = {}
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, torch.Tensor):
                value = value.to(device=device, non_blocking=non_blocking)
            values[field.name] = value
        return CanonicalBatch(**values)

    @property
    def batch_size(self) -> int:
        return int(self.future_state.shape[0])

    @property
    def history_valid_mask(self) -> torch.Tensor:
        return self.history_query_mask

    @property
    def future_valid_mask(self) -> torch.Tensor:
        return self.future_query_stream_mask.any(dim=-1)

    @property
    def history_padding_mask(self) -> torch.Tensor:
        """PyTorch attention convention: True means padding/ignore."""
        return ~self.history_valid_mask

    @property
    def future_padding_mask(self) -> torch.Tensor:
        """PyTorch attention convention: True means padding/ignore."""
        return ~self.future_valid_mask

    def validate(self) -> None:
        if self.history_state.ndim != 3 or self.history_state.shape[-1] != GEOMETRY_DIM:
            raise ValueError(f"history_state must be [B,H,{GEOMETRY_DIM}]")
        if self.future_state.ndim != 3 or self.future_state.shape[-1] != GEOMETRY_DIM:
            raise ValueError(f"future_state must be [B,F,{GEOMETRY_DIM}]")
        if self.history_time.shape != self.history_state.shape[:2]:
            raise ValueError("history_time must match history_state [B,H]")
        if self.future_time.shape != self.future_state.shape[:2]:
            raise ValueError("future_time must match future_state [B,F]")
        if self.history_query_mask.shape != self.history_time.shape:
            raise ValueError("history_query_mask must match history_time [B,H]")
        expected_history_mask = (*self.history_state.shape[:2], len(STREAM_NAMES))
        expected_future_mask = (*self.future_state.shape[:2], len(STREAM_NAMES))
        if self.history_stream_mask.shape != expected_history_mask:
            raise ValueError(f"history_stream_mask must be {expected_history_mask}")
        if self.future_stream_mask.shape != expected_future_mask:
            raise ValueError(f"future_stream_mask must be {expected_future_mask}")
        if self.future_query_stream_mask.shape != expected_future_mask:
            raise ValueError(f"future_query_stream_mask must be {expected_future_mask}")
        boolean_masks = (
            self.history_query_mask,
            self.history_stream_mask,
            self.future_query_stream_mask,
            self.future_stream_mask,
        )
        if any(mask.dtype != torch.bool for mask in boolean_masks):
            raise TypeError("query and stream masks must use torch.bool")
        if (self.future_stream_mask & ~self.future_query_stream_mask).any():
            raise ValueError("Supervision cannot be valid for a disabled future query stream")
        for name, tensor in (
            ("history_time", self.history_time),
            ("history_state", self.history_state),
            ("future_time", self.future_time),
            ("future_state", self.future_state),
        ):
            if not torch.isfinite(tensor).all():
                raise ValueError(f"{name} contains NaN or Inf")
        if self.intrinsics.shape != (self.batch_size, 4):
            raise ValueError("intrinsics must be normalized [B,4] = fx/W,fy/H,cx/W,cy/H")
        if len(self.text) != self.batch_size:
            raise ValueError("text length must match batch size")
        if self.context_images is not None:
            if self.context_images.ndim != 5 or self.context_images.shape[:2] != self.history_time.shape:
                raise ValueError("context_images must be [B,H,3,H_img,W_img]")
            if self.context_images.shape[2] != 3:
                raise ValueError("context_images must have three RGB channels")
        if self.context_visual_features is not None:
            if self.context_visual_features.shape[:2] != self.history_time.shape:
                raise ValueError("context_visual_features must align with [B,H]")
        if self.future_visual_latents is not None:
            if self.future_visual_latents.shape[:2] != self.future_time.shape:
                raise ValueError("future_visual_latents must align with [B,F]")
        if self.history_fingertips is not None:
            expected = (*self.history_time.shape, 2, 5, 3)
            if self.history_fingertips.shape != expected:
                raise ValueError(f"history_fingertips must be {expected}")
        if self.future_fingertips is not None:
            expected = (*self.future_time.shape, 2, 5, 3)
            if self.future_fingertips.shape != expected:
                raise ValueError(f"future_fingertips must be {expected}")


def _pad_temporal(samples: list[dict[str, Any]], key: str, value: float | bool = 0) -> torch.Tensor:
    tensors = [sample[key] for sample in samples]
    if not all(isinstance(tensor, torch.Tensor) for tensor in tensors):
        raise TypeError(f"Every {key} value must be a tensor")
    return pad_sequence(tensors, batch_first=True, padding_value=value)


def _optional_pad(samples: list[dict[str, Any]], key: str) -> torch.Tensor | None:
    values = [sample.get(key) for sample in samples]
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise ValueError(f"Mixed presence for optional field {key}; split datasets or provide masks")
    return pad_sequence(values, batch_first=True, padding_value=0.0)


def canonical_collate(samples: Iterable[dict[str, Any]]) -> CanonicalBatch:
    samples = list(samples)
    if not samples:
        raise ValueError("Cannot collate an empty batch")
    batch = CanonicalBatch(
        history_time=_pad_temporal(samples, "history_time", 0.0),
        history_query_mask=_pad_temporal(samples, "history_query_mask", False).bool(),
        history_state=_pad_temporal(samples, "history_state", 0.0),
        history_stream_mask=_pad_temporal(samples, "history_stream_mask", False).bool(),
        future_time=_pad_temporal(samples, "future_time", 0.0),
        future_query_stream_mask=_pad_temporal(
            samples, "future_query_stream_mask", False
        ).bool(),
        future_state=_pad_temporal(samples, "future_state", 0.0),
        future_stream_mask=_pad_temporal(samples, "future_stream_mask", False).bool(),
        text=[str(sample.get("text", "")) for sample in samples],
        intrinsics=torch.stack([sample["intrinsics"] for sample in samples]),
        context_images=_optional_pad(samples, "context_images"),
        context_visual_features=_optional_pad(samples, "context_visual_features"),
        future_visual_latents=_optional_pad(samples, "future_visual_latents"),
        history_fingertips=_optional_pad(samples, "history_fingertips"),
        future_fingertips=_optional_pad(samples, "future_fingertips"),
        metadata=[sample.get("metadata", {}) for sample in samples],
    )
    batch.validate()
    return batch
