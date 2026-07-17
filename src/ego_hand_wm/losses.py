"""Masked flow and decoded geometric objectives."""

from __future__ import annotations

from typing import Any

import torch
import torch.distributed as distributed
import torch.nn.functional as functional

from ego_hand_wm.contracts.batch import CanonicalBatch
from ego_hand_wm.contracts.schema import SCHEMA, STREAM_NAMES
from ego_hand_wm.geometry.rotations import rotation_6d_to_matrix, so3_geodesic_angle
from ego_hand_wm.geometry.se3 import invert, pose9_to_matrix


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(value.dtype)
    numerator = (value * mask).sum()
    count = mask.sum()
    if not (distributed.is_available() and distributed.is_initialized()):
        return numerator / count.clamp_min(1.0)

    global_count = count.detach().clone()
    global_numerator = numerator.detach().clone()
    distributed.all_reduce(global_count, op=distributed.ReduceOp.SUM)
    distributed.all_reduce(global_numerator, op=distributed.ReduceOp.SUM)
    denominator = global_count.clamp_min(1.0)
    # DDP averages gradients across ranks. Scaling each local numerator by world-size gives the
    # gradient of the true global masked mean; the detached correction makes logging identical.
    local_objective = numerator * distributed.get_world_size() / denominator
    global_value = global_numerator / denominator
    return local_objective + (global_value - local_objective.detach())


def masked_stream_flow_loss(
    prediction: torch.Tensor, target: torch.Tensor, stream_mask: torch.Tensor
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    split_prediction = SCHEMA.split(prediction)
    split_target = SCHEMA.split(target)
    losses: dict[str, torch.Tensor] = {}
    for index, name in enumerate(STREAM_NAMES):
        per_token = (split_prediction[name] - split_target[name]).square().mean(dim=-1)
        losses[f"flow/{name}"] = _masked_mean(per_token, stream_mask[..., index])
    return torch.stack(tuple(losses.values())).sum(), losses


def rotation_geodesic_loss(
    prediction: torch.Tensor, target: torch.Tensor, stream_mask: torch.Tensor
) -> torch.Tensor:
    pred = SCHEMA.split(prediction)
    gt = SCHEMA.split(target)
    terms: list[torch.Tensor] = []
    for index, name in enumerate(("camera", "left_wrist", "right_wrist")):
        pred_rotation = rotation_6d_to_matrix(pred[name][..., 3:9])
        gt_rotation = rotation_6d_to_matrix(gt[name][..., 3:9])
        terms.append(
            _masked_mean(so3_geodesic_angle(pred_rotation, gt_rotation), stream_mask[..., index])
        )
    for index, name in ((3, "left_mano"), (4, "right_mano")):
        pred_rotation = rotation_6d_to_matrix(pred[name].reshape(*pred[name].shape[:-1], 15, 6))
        gt_rotation = rotation_6d_to_matrix(gt[name].reshape(*gt[name].shape[:-1], 15, 6))
        per_token = so3_geodesic_angle(pred_rotation, gt_rotation).mean(dim=-1)
        terms.append(_masked_mean(per_token, stream_mask[..., index]))
    return torch.stack(terms).mean()


def camera_hand_decomposition_loss(
    prediction: torch.Tensor, target: torch.Tensor, stream_mask: torch.Tensor
) -> torch.Tensor:
    pred = SCHEMA.split(prediction)
    gt = SCHEMA.split(target)
    pred_camera = pose9_to_matrix(pred["camera"])
    gt_camera = pose9_to_matrix(gt["camera"])
    terms: list[torch.Tensor] = []
    for hand_index, name in ((1, "left_wrist"), (2, "right_wrist")):
        pred_relative = invert(pred_camera) @ pose9_to_matrix(pred[name])
        gt_relative = invert(gt_camera) @ pose9_to_matrix(gt[name])
        translation = functional.smooth_l1_loss(
            pred_relative[..., :3, 3], gt_relative[..., :3, 3], reduction="none"
        ).mean(dim=-1)
        rotation = so3_geodesic_angle(
            pred_relative[..., :3, :3], gt_relative[..., :3, :3]
        )
        valid = stream_mask[..., 0] & stream_mask[..., hand_index]
        terms.append(_masked_mean(translation + rotation, valid))
    return torch.stack(terms).mean()


class WorldActionLoss(torch.nn.Module):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.rotation_weight = float(config.get("rotation_weight", 0.1))
        self.ego_weight = float(config.get("ego_weight", 0.2))
        self.visual_weight = float(config.get("visual_weight", 0.0))

    def forward(
        self,
        batch: CanonicalBatch,
        predicted_velocity: torch.Tensor,
        target_velocity: torch.Tensor,
        noisy_state: torch.Tensor,
        flow_time: torch.Tensor,
        *,
        predicted_visual_velocity: torch.Tensor | None = None,
        target_visual_velocity: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        flow, components = masked_stream_flow_loss(
            predicted_velocity, target_velocity, batch.future_stream_mask
        )
        estimated_clean = noisy_state + (1.0 - flow_time[:, None, None]) * predicted_velocity
        rotation = rotation_geodesic_loss(
            estimated_clean, batch.future_state, batch.future_stream_mask
        )
        ego = camera_hand_decomposition_loss(
            estimated_clean, batch.future_state, batch.future_stream_mask
        )
        total = flow + self.rotation_weight * rotation + self.ego_weight * ego
        output = {"loss": total, "flow": flow, "rotation": rotation, "ego": ego, **components}
        if predicted_visual_velocity is not None and target_visual_velocity is not None:
            visual_per_token = (
                predicted_visual_velocity - target_visual_velocity
            ).square().mean(dim=-1)
            visual = _masked_mean(visual_per_token, batch.future_valid_mask)
            output["visual"] = visual
            output["loss"] = output["loss"] + self.visual_weight * visual
        return output
