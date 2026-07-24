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

# MANO orders its 15 articulated rotations as five contiguous three-joint chains.  Keeping
# each chain in one token gives the transformer an anatomical unit without changing the
# canonical 207-D storage contract used by every dataset adapter.
MANO_CHAIN_NAMES = ("index", "middle", "pinky", "ring", "thumb")
MANO_CHAIN_DIM = 3 * ROT6D_DIM
ENTITY_NAMES = (
    "camera",
    "left_wrist",
    "right_wrist",
    *(f"left_{name}" for name in MANO_CHAIN_NAMES),
    *(f"right_{name}" for name in MANO_CHAIN_NAMES),
)
ENTITY_DIMS = (CAMERA_DIM, WRIST_DIM, WRIST_DIM) + (MANO_CHAIN_DIM,) * 10
ENTITY_STREAM_INDICES = (0, 1, 2) + (3,) * 5 + (4,) * 5


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

    def split_entities(self, state: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Split canonical state into camera, wrists, and ten MANO finger-chain entities."""
        streams = self.split(state)
        left = streams["left_mano"].reshape(*state.shape[:-1], 5, MANO_CHAIN_DIM)
        right = streams["right_mano"].reshape(*state.shape[:-1], 5, MANO_CHAIN_DIM)
        return (
            streams["camera"],
            streams["left_wrist"],
            streams["right_wrist"],
            *left.unbind(dim=-2),
            *right.unbind(dim=-2),
        )

    def pack_entities(self, entities: tuple[torch.Tensor, ...] | list[torch.Tensor]) -> torch.Tensor:
        """Invert :meth:`split_entities` while preserving the canonical stream order."""
        if len(entities) != len(ENTITY_NAMES):
            raise ValueError(f"Expected {len(ENTITY_NAMES)} entities, got {len(entities)}")
        for name, entity, expected in zip(ENTITY_NAMES, entities, ENTITY_DIMS, strict=True):
            if entity.shape[-1] != expected:
                raise ValueError(f"{name} must have final dim {expected}, got {entity.shape[-1]}")
        left_mano = torch.cat(tuple(entities[3:8]), dim=-1)
        right_mano = torch.cat(tuple(entities[8:13]), dim=-1)
        return self.pack(entities[0], entities[1], entities[2], left_mano, right_mano)

    def expand_entity_mask(self, stream_mask: torch.Tensor) -> torch.Tensor:
        """Map five canonical stream-validity flags to the 13 anatomical entities."""
        if stream_mask.shape[-1] != len(STREAM_NAMES):
            raise ValueError(
                f"Expected {len(STREAM_NAMES)} stream masks, got {stream_mask.shape[-1]}"
            )
        indices = torch.tensor(ENTITY_STREAM_INDICES, device=stream_mask.device)
        return stream_mask.index_select(-1, indices)


SCHEMA = GeometrySchema()
