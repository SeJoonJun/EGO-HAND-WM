"""Deterministic smooth canonical trajectories for CPU and distributed smoke tests."""

from __future__ import annotations

import torch
from torch.utils.data import Dataset

from ego_hand_wm.contracts.schema import SCHEMA
from ego_hand_wm.geometry.rotations import axis_angle_to_matrix, matrix_to_rotation_6d


def _smooth_rot6(
    generator: torch.Generator, times: torch.Tensor, entities: int, scale: float
) -> torch.Tensor:
    axes = torch.randn(entities, 3, generator=generator)
    axes = axes / axes.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    speeds = torch.randn(entities, 1, generator=generator) * scale
    vectors = times[:, None, None] * axes[None] * speeds[None]
    return matrix_to_rotation_6d(axis_angle_to_matrix(vectors))


class SyntheticCanonicalDataset(Dataset):
    provides_context_visual = True
    provides_future_visual = False

    def __init__(
        self,
        length: int = 64,
        history_steps: int = 3,
        future_steps: int = 4,
        horizon_seconds: float = 1.0,
        image_size: int = 32,
        seed: int = 17,
    ) -> None:
        self.length = int(length)
        self.history_steps = int(history_steps)
        self.future_steps = int(future_steps)
        self.horizon_seconds = float(horizon_seconds)
        self.image_size = int(image_size)
        self.seed = int(seed)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str | dict[str, int]]:
        generator = torch.Generator().manual_seed(self.seed + int(index))
        history_time = torch.linspace(-0.5, 0.0, self.history_steps)
        future_time = torch.linspace(
            self.horizon_seconds / self.future_steps,
            self.horizon_seconds,
            self.future_steps,
        )
        all_time = torch.cat((history_time, future_time))
        count = all_time.numel()

        camera_translation = torch.zeros(count, 3)
        camera_translation[:, 0] = 0.05 * all_time
        camera_translation[:, 1] = 0.01 * torch.sin(2.0 * all_time)
        camera_rotation = _smooth_rot6(generator, all_time, 1, 0.12)[:, 0]
        camera = torch.cat((camera_translation, camera_rotation), dim=-1)

        wrist_parts = []
        for hand_sign in (-1.0, 1.0):
            base = torch.tensor([0.22 * hand_sign, -0.18, 0.55])
            velocity = torch.randn(3, generator=generator) * 0.04
            translation = base + all_time[:, None] * velocity
            rotation = _smooth_rot6(generator, all_time, 1, 0.35)[:, 0]
            wrist_parts.append(torch.cat((translation, rotation), dim=-1))

        hand_parts = []
        for _ in range(2):
            rotations = _smooth_rot6(generator, all_time, 15, 0.6)
            hand_parts.append(rotations.reshape(count, 90))

        state = SCHEMA.pack(camera, wrist_parts[0], wrist_parts[1], hand_parts[0], hand_parts[1])
        stream_mask = torch.ones(count, 5, dtype=torch.bool)
        if index % 7 == 0:
            stream_mask[:, 1] = False
            stream_mask[:, 3] = False

        images = torch.rand(
            self.history_steps, 3, self.image_size, self.image_size, generator=generator
        )
        return {
            "history_time": history_time,
            "history_query_mask": torch.ones(self.history_steps, dtype=torch.bool),
            "history_state": state[: self.history_steps],
            "history_stream_mask": stream_mask[: self.history_steps],
            "future_time": future_time,
            "future_query_stream_mask": torch.ones(
                self.future_steps, 5, dtype=torch.bool
            ),
            "future_state": state[self.history_steps :],
            "future_stream_mask": stream_mask[self.history_steps :],
            "text": "move both hands toward the object",
            "intrinsics": torch.tensor([0.8, 0.8, 0.5, 0.5]),
            "context_images": images,
            "metadata": {"synthetic_index": int(index)},
        }
