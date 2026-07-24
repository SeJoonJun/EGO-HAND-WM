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


FINGERTIP_INDICES = (4, 8, 12, 16, 20)


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
    prediction: torch.Tensor,
    target: torch.Tensor,
    stream_mask: torch.Tensor,
    state_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    split_prediction = SCHEMA.split(prediction)
    split_target = SCHEMA.split(target)
    split_state_mask = (
        SCHEMA.split(SCHEMA.expand_stream_mask(stream_mask))
        if state_mask is None
        else SCHEMA.split(state_mask)
    )
    losses: dict[str, torch.Tensor] = {}
    for index, name in enumerate(STREAM_NAMES):
        del index
        squared_error = (split_prediction[name] - split_target[name]).square()
        losses[f"flow/{name}"] = _masked_mean(
            squared_error, split_state_mask[name]
        )
    return torch.stack(tuple(losses.values())).sum(), losses


def rotation_geodesic_loss(
    prediction: torch.Tensor, target: torch.Tensor, stream_mask: torch.Tensor
) -> torch.Tensor:
    rigid, mano, all_terms = rotation_geodesic_components(
        prediction, target, stream_mask
    )
    del rigid, mano
    return all_terms


def rotation_geodesic_components(
    prediction: torch.Tensor,
    target: torch.Tensor,
    stream_mask: torch.Tensor,
    state_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return rigid, MANO, and legacy five-stream geodesic means.

    The original objective averaged camera, two wrists, and two MANO hands before applying one
    small weight.  Exposing the groups lets a refinement run strengthen articulation without
    perturbing the already-good rigid-pose branch.  The third value exactly preserves the old
    objective for existing configurations.
    """
    pred = SCHEMA.split(prediction)
    gt = SCHEMA.split(target)
    component = (
        SCHEMA.split(SCHEMA.expand_stream_mask(stream_mask))
        if state_mask is None
        else SCHEMA.split(state_mask)
    )
    rigid_terms: list[torch.Tensor] = []
    for index, name in enumerate(("camera", "left_wrist", "right_wrist")):
        pred_rotation = rotation_6d_to_matrix(pred[name][..., 3:9])
        gt_rotation = rotation_6d_to_matrix(gt[name][..., 3:9])
        rotation_valid = component[name][..., 3:9].all(dim=-1)
        rigid_terms.append(
            _masked_mean(
                so3_geodesic_angle(pred_rotation, gt_rotation),
                stream_mask[..., index] & rotation_valid,
            )
        )
    mano_terms: list[torch.Tensor] = []
    for index, name in ((3, "left_mano"), (4, "right_mano")):
        pred_rotation = rotation_6d_to_matrix(pred[name].reshape(*pred[name].shape[:-1], 15, 6))
        gt_rotation = rotation_6d_to_matrix(gt[name].reshape(*gt[name].shape[:-1], 15, 6))
        per_token = so3_geodesic_angle(pred_rotation, gt_rotation).mean(dim=-1)
        joint_valid = component[name].reshape(*component[name].shape[:-1], 15, 6).all(dim=-1)
        mano_terms.append(
            _masked_mean(
                so3_geodesic_angle(pred_rotation, gt_rotation),
                stream_mask[..., index, None] & joint_valid,
            )
        )
    rigid = torch.stack(rigid_terms).mean()
    mano = torch.stack(mano_terms).mean()
    legacy = torch.stack((*rigid_terms, *mano_terms)).mean()
    return rigid, mano, legacy


def hand_kinematics_losses(
    batch: CanonicalBatch,
    prediction: torch.Tensor,
    *,
    position_beta: float,
    motion_beta: float,
) -> dict[str, torch.Tensor]:
    """Wrist-local full-joint, fingertip, and physical-velocity objectives."""
    target = batch.future_hand_joints_local
    history = batch.history_hand_joints_local
    if target is None or history is None:
        raise ValueError("Hand-kinematics loss requires local history and future 21-joint targets")
    if prediction.shape != target.shape:
        raise ValueError(
            f"Predicted hand joints {tuple(prediction.shape)} do not match {tuple(target.shape)}"
        )
    mask = batch.future_stream_mask[..., [3, 4]]
    point_error = functional.smooth_l1_loss(
        prediction.float(), target.float(), beta=position_beta, reduction="none"
    ).mean(dim=-1)
    joint = _masked_mean(point_error, mask[..., None].expand_as(point_error))
    fingertip = _masked_mean(
        point_error[..., FINGERTIP_INDICES],
        mask[..., None].expand_as(point_error[..., FINGERTIP_INDICES]),
    )

    # Match physical motion even when future samples are not consecutive video frames.  The
    # anchor is exactly t=0, and each later velocity uses its actual timestamp interval.
    previous_prediction = torch.cat((history[:, -1:, :, :, :], prediction[:, :-1]), dim=1)
    previous_target = torch.cat((history[:, -1:, :, :, :], target[:, :-1]), dim=1)
    previous_time = torch.cat(
        (torch.zeros_like(batch.future_time[:, :1]), batch.future_time[:, :-1]), dim=1
    )
    delta_time = (batch.future_time - previous_time).clamp_min(1e-4)
    predicted_velocity = (prediction - previous_prediction) / delta_time[..., None, None, None]
    target_velocity = (target - previous_target) / delta_time[..., None, None, None]
    previous_mask = torch.cat(
        (batch.history_stream_mask[:, -1:, [3, 4]], mask[:, :-1]), dim=1
    )
    motion_mask = mask & previous_mask
    motion_error = functional.smooth_l1_loss(
        predicted_velocity.float(),
        target_velocity.float(),
        beta=motion_beta,
        reduction="none",
    ).mean(dim=-1)
    motion = _masked_mean(
        motion_error, motion_mask[..., None].expand_as(motion_error)
    )
    return {"hand_joint": joint, "fingertip": fingertip, "hand_motion": motion}


def camera_hand_decomposition_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    stream_mask: torch.Tensor,
    state_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    pred = SCHEMA.split(prediction)
    gt = SCHEMA.split(target)
    pred_camera = pose9_to_matrix(pred["camera"])
    gt_camera = pose9_to_matrix(gt["camera"])
    component = (
        SCHEMA.split(SCHEMA.expand_stream_mask(stream_mask))
        if state_mask is None
        else SCHEMA.split(state_mask)
    )
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
        camera_pose_valid = component["camera"].all(dim=-1)
        translation_valid = (
            camera_pose_valid
            & component[name][..., :3].all(dim=-1)
            & stream_mask[..., 0]
            & stream_mask[..., hand_index]
        )
        rotation_valid = (
            camera_pose_valid
            & component[name][..., 3:9].all(dim=-1)
            & stream_mask[..., 0]
            & stream_mask[..., hand_index]
        )
        terms.append(
            _masked_mean(translation, translation_valid)
            + _masked_mean(rotation, rotation_valid)
        )
    return torch.stack(terms).mean()


def wrist_trajectory_losses(
    batch: CanonicalBatch,
    estimated_clean: torch.Tensor,
    *,
    horizon_power: float,
    velocity_beta: float,
) -> dict[str, torch.Tensor]:
    """Metric-aligned wrist XYZ losses for sparse trajectory fine-tuning.

    Rectified-flow MSE treats every state coordinate equally and can become numerically tiny
    when translations are measured in metres.  H2O/EgoPAT3D, however, are evaluated directly
    with wrist ADE/FDE.  These objectives supervise the clean-state estimate implied by the
    velocity field at the sampled flow time; they neither copy nor add the last observation to
    the prediction.

    ``horizon_power`` optionally emphasizes later valid targets.  The separate endpoint term is
    always evaluated at the final valid target of each hand, so variable-length VITRA episodes
    and padded batches remain well defined.
    """

    if horizon_power < 0:
        raise ValueError("trajectory_horizon_power must be non-negative")
    if velocity_beta <= 0:
        raise ValueError("trajectory_velocity_beta must be positive")

    prediction = SCHEMA.split(estimated_clean)
    target = SCHEMA.split(batch.future_state)
    component = SCHEMA.split(batch.effective_future_state_mask)
    history = SCHEMA.split(batch.history_state)
    history_component = SCHEMA.split(batch.effective_history_state_mask)

    position_errors: list[torch.Tensor] = []
    position_weights: list[torch.Tensor] = []
    endpoint_errors: list[torch.Tensor] = []
    endpoint_masks: list[torch.Tensor] = []
    velocity_errors: list[torch.Tensor] = []
    velocity_masks: list[torch.Tensor] = []

    for stream_index, name in ((1, "left_wrist"), (2, "right_wrist")):
        valid = (
            batch.future_stream_mask[..., stream_index]
            & component[name][..., :3].all(dim=-1)
        )
        delta = prediction[name][..., :3] - target[name][..., :3]
        position_error = torch.linalg.vector_norm(delta.float(), dim=-1)

        if horizon_power == 0:
            horizon_weight = torch.ones_like(position_error)
        else:
            valid_time = torch.where(valid, batch.future_time, torch.zeros_like(batch.future_time))
            final_time = valid_time.amax(dim=1, keepdim=True).clamp_min(1e-6)
            horizon_weight = (batch.future_time / final_time).clamp(0.0, 1.0).pow(
                horizon_power
            )
        position_errors.append(position_error)
        position_weights.append(valid.to(position_error.dtype) * horizon_weight)

        # A reverse cumulative count equals one only at the final valid element, including when
        # masks contain gaps or sequences are padded.
        remaining = torch.flip(
            torch.cumsum(torch.flip(valid.to(torch.int64), dims=(1,)), dim=1),
            dims=(1,),
        )
        endpoint_errors.append(position_error)
        endpoint_masks.append(valid & remaining.eq(1))

        history_valid = (
            batch.history_stream_mask[..., stream_index]
            & history_component[name][..., :3].all(dim=-1)
        )
        history_indices = torch.arange(
            batch.history_state.shape[1], device=estimated_clean.device
        ).view(1, -1)
        last_history_index = torch.where(
            history_valid, history_indices, torch.full_like(history_indices, -1)
        ).amax(dim=1)
        safe_history_index = last_history_index.clamp_min(0)
        gather_index = safe_history_index[:, None, None].expand(-1, 1, 3)
        anchor = history[name][..., :3].gather(1, gather_index)

        predicted_sequence = torch.cat((anchor, prediction[name][..., :3]), dim=1)
        target_sequence = torch.cat((anchor, target[name][..., :3]), dim=1)
        delta_time = torch.cat(
            (
                batch.future_time[:, :1],
                batch.future_time[:, 1:] - batch.future_time[:, :-1],
            ),
            dim=1,
        ).clamp_min(1e-4)
        predicted_velocity = (
            predicted_sequence[:, 1:] - predicted_sequence[:, :-1]
        ) / delta_time[..., None]
        target_velocity = (
            target_sequence[:, 1:] - target_sequence[:, :-1]
        ) / delta_time[..., None]
        velocity_error = functional.smooth_l1_loss(
            predicted_velocity.float(),
            target_velocity.float(),
            beta=velocity_beta,
            reduction="none",
        ).mean(dim=-1)
        previous_valid = torch.cat(
            (last_history_index.ge(0)[:, None], valid[:, :-1]), dim=1
        )
        velocity_errors.append(velocity_error)
        velocity_masks.append(valid & previous_valid)

    all_position_error = torch.stack(position_errors, dim=-1)
    all_position_weight = torch.stack(position_weights, dim=-1)
    all_endpoint_error = torch.stack(endpoint_errors, dim=-1)
    all_endpoint_mask = torch.stack(endpoint_masks, dim=-1)
    all_velocity_error = torch.stack(velocity_errors, dim=-1)
    all_velocity_mask = torch.stack(velocity_masks, dim=-1)
    return {
        "trajectory/position": _masked_mean(
            all_position_error, all_position_weight
        ),
        "trajectory/endpoint": _masked_mean(
            all_endpoint_error, all_endpoint_mask
        ),
        "trajectory/velocity": _masked_mean(
            all_velocity_error, all_velocity_mask
        ),
    }


class WorldActionLoss(torch.nn.Module):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.rotation_weight = float(config.get("rotation_weight", 0.1))
        self.use_factorized_rotation = (
            "rigid_rotation_weight" in config or "mano_rotation_weight" in config
        )
        self.rigid_rotation_weight = float(
            config.get("rigid_rotation_weight", self.rotation_weight)
        )
        self.mano_rotation_weight = float(
            config.get("mano_rotation_weight", self.rotation_weight)
        )
        self.ego_weight = float(config.get("ego_weight", 0.2))
        self.visual_weight = float(config.get("visual_weight", 0.0))
        self.hand_joint_weight = float(config.get("hand_joint_weight", 0.0))
        self.fingertip_weight = float(config.get("fingertip_weight", 0.0))
        self.hand_motion_weight = float(config.get("hand_motion_weight", 0.0))
        self.hand_position_beta = float(config.get("hand_position_beta", 0.01))
        self.hand_motion_beta = float(config.get("hand_motion_beta", 0.05))
        self.trajectory_position_weight = float(
            config.get("trajectory_position_weight", 0.0)
        )
        self.trajectory_endpoint_weight = float(
            config.get("trajectory_endpoint_weight", 0.0)
        )
        self.trajectory_velocity_weight = float(
            config.get("trajectory_velocity_weight", 0.0)
        )
        self.trajectory_horizon_power = float(
            config.get("trajectory_horizon_power", 0.0)
        )
        self.trajectory_velocity_beta = float(
            config.get("trajectory_velocity_beta", 0.05)
        )
        if self.hand_position_beta <= 0 or self.hand_motion_beta <= 0:
            raise ValueError("Hand Smooth-L1 beta values must be positive")
        if min(
            self.trajectory_position_weight,
            self.trajectory_endpoint_weight,
            self.trajectory_velocity_weight,
            self.trajectory_horizon_power,
        ) < 0:
            raise ValueError("Trajectory loss weights and horizon power must be non-negative")
        if self.trajectory_velocity_beta <= 0:
            raise ValueError("trajectory_velocity_beta must be positive")

    @property
    def requires_hand_kinematics(self) -> bool:
        return any(
            weight > 0
            for weight in (
                self.hand_joint_weight,
                self.fingertip_weight,
                self.hand_motion_weight,
            )
        )

    @property
    def requires_trajectory_losses(self) -> bool:
        return any(
            weight > 0
            for weight in (
                self.trajectory_position_weight,
                self.trajectory_endpoint_weight,
                self.trajectory_velocity_weight,
            )
        )

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
        predicted_hand_joints_local: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        flow, components = masked_stream_flow_loss(
            predicted_velocity,
            target_velocity,
            batch.future_stream_mask,
            batch.effective_future_state_mask,
        )
        estimated_clean = noisy_state + (1.0 - flow_time[:, None, None]) * predicted_velocity
        rigid_rotation, mano_rotation, rotation = rotation_geodesic_components(
            estimated_clean,
            batch.future_state,
            batch.future_stream_mask,
            batch.effective_future_state_mask,
        )
        ego = camera_hand_decomposition_loss(
            estimated_clean,
            batch.future_state,
            batch.future_stream_mask,
            batch.effective_future_state_mask,
        )
        if self.use_factorized_rotation:
            rotation_objective = (
                self.rigid_rotation_weight * rigid_rotation
                + self.mano_rotation_weight * mano_rotation
            )
        else:
            rotation_objective = self.rotation_weight * rotation
        total = flow + rotation_objective + self.ego_weight * ego
        output = {
            "loss": total,
            "flow": flow,
            "rotation": rotation,
            "rotation/rigid": rigid_rotation,
            "rotation/mano": mano_rotation,
            "ego": ego,
            **components,
        }
        if self.requires_trajectory_losses:
            trajectory = wrist_trajectory_losses(
                batch,
                estimated_clean,
                horizon_power=self.trajectory_horizon_power,
                velocity_beta=self.trajectory_velocity_beta,
            )
            output.update(trajectory)
            output["loss"] = (
                output["loss"]
                + self.trajectory_position_weight * trajectory["trajectory/position"]
                + self.trajectory_endpoint_weight * trajectory["trajectory/endpoint"]
                + self.trajectory_velocity_weight * trajectory["trajectory/velocity"]
            )
        if predicted_visual_velocity is not None and target_visual_velocity is not None:
            visual_per_token = (
                predicted_visual_velocity - target_visual_velocity
            ).square().mean(dim=-1)
            visual_mask = batch.future_valid_mask
            while visual_mask.ndim < visual_per_token.ndim:
                visual_mask = visual_mask.unsqueeze(-1)
            visual = _masked_mean(visual_per_token, visual_mask.expand_as(visual_per_token))
            output["visual"] = visual
            output["loss"] = output["loss"] + self.visual_weight * visual
        if predicted_hand_joints_local is not None:
            hand_losses = hand_kinematics_losses(
                batch,
                predicted_hand_joints_local,
                position_beta=self.hand_position_beta,
                motion_beta=self.hand_motion_beta,
            )
            output.update(hand_losses)
            output["loss"] = (
                output["loss"]
                + self.hand_joint_weight * hand_losses["hand_joint"]
                + self.fingertip_weight * hand_losses["fingertip"]
                + self.hand_motion_weight * hand_losses["hand_motion"]
            )
        elif self.requires_hand_kinematics:
            raise ValueError(
                "A hand-kinematics loss is enabled, but the model produced no hand joints"
            )
        return output
