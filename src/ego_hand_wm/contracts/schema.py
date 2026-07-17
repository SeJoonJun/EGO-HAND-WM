"""The one model-facing geometry layout used by every dataset and embodiment."""

from __future__ import annotations

from typing import ClassVar

import torch

CAMERA_DIM = 9
WRIST_DIM = 9
MANO_JOINTS = 15
ROT6D_DIM = 6
HAND_DIM = MANO_JOINTS * ROT6D_DIM
GEOMETRY_DIM = CAMERA_DIM + 2 * WRIST_DIM + 2 * HAND_DIM

STREAM_NAMES = ("camera", "left_wrist", "right_wrist", "left_mano", "right_mano")
STREAM_DIMS = (CAMERA_DIM, WRIST_DIM, WRIST_DIM, HAND_DIM, HAND_DIM)


class GeometrySchema:
    """Slices and packing helpers for the canonical 207-D future state."""

    camera: ClassVar[slice] = slice(0, 9)
    left_wrist: ClassVar[slice] = slice(9, 18)
    right_wrist: ClassVar[slice] = slice(18, 27)
    left_mano: ClassVar[slice] = slice(27, 117)
    right_mano: ClassVar[slice] = slice(117, 207)

    @property
    def stream_slices(self) -> tuple[slice, ...]:
        return (
            self.camera,
            self.left_wrist,
            self.right_wrist,
            self.left_mano,
            self.right_mano,
        )

    def split(self, state: torch.Tensor) -> dict[str, torch.Tensor]:
        if state.shape[-1] != GEOMETRY_DIM:
            raise ValueError(f"Expected state dim {GEOMETRY_DIM}, got {state.shape[-1]}")
        return {
            name: state[..., stream_slice]
            for name, stream_slice in zip(STREAM_NAMES, self.stream_slices, strict=True)
        }

    def pack(
        self,
        camera: torch.Tensor,
        left_wrist: torch.Tensor,
        right_wrist: torch.Tensor,
        left_mano: torch.Tensor,
        right_mano: torch.Tensor,
    ) -> torch.Tensor:
        parts = (camera, left_wrist, right_wrist, left_mano, right_mano)
        for name, part, expected in zip(STREAM_NAMES, parts, STREAM_DIMS, strict=True):
            if part.shape[-1] != expected:
                raise ValueError(f"{name} must have final dim {expected}, got {part.shape[-1]}")
        return torch.cat(parts, dim=-1)

    def expand_stream_mask(self, stream_mask: torch.Tensor) -> torch.Tensor:
        if stream_mask.shape[-1] != len(STREAM_NAMES):
            raise ValueError(
                f"Expected {len(STREAM_NAMES)} stream masks, got {stream_mask.shape[-1]}"
            )
        expanded = [
            stream_mask[..., index : index + 1].expand(*stream_mask.shape[:-1], width)
            for index, width in enumerate(STREAM_DIMS)
        ]
        return torch.cat(expanded, dim=-1)


SCHEMA = GeometrySchema()
