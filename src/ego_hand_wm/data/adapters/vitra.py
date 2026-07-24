"""VITRA raw episode to canonical A=C0 geometry conversion.

This module intentionally never imports VITRA's training package. It reads the released raw
episode schema and emits the shared contract. VITRA stores both sides in a MANO_RIGHT-derived
local-pose convention; ``as_stored`` preserves that released, self-consistent representation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np
import torch

from ego_hand_wm.contracts.schema import SCHEMA
from ego_hand_wm.geometry.rotations import matrix_to_rotation_6d
from ego_hand_wm.geometry.se3 import encode_pose9, invert, transform_points

LeftManoPolicy = Literal["mask", "mirror_x", "as_stored"]
FINGERTIP_INDICES = (4, 8, 12, 16, 20)


def load_vitra_episode(path: str | Path) -> dict[str, Any]:
    episode = np.load(Path(path), allow_pickle=True).item()
    if not isinstance(episode, dict):
        raise ValueError(f"VITRA annotation {path} did not contain a dictionary")
    return episode


def _as_float_tensor(value: Any) -> torch.Tensor:
    return torch.as_tensor(np.asarray(value), dtype=torch.float32)


def _make_transform(rotation: torch.Tensor, translation: torch.Tensor) -> torch.Tensor:
    result = torch.zeros(*rotation.shape[:-2], 4, 4, dtype=rotation.dtype)
    result[..., :3, :3] = rotation
    result[..., :3, 3] = translation
    result[..., 3, 3] = 1.0
    return result


def _mirror_rotations_x(rotation: torch.Tensor) -> torch.Tensor:
    reflection = torch.diag(torch.tensor([-1.0, 1.0, 1.0], dtype=rotation.dtype))
    return reflection @ rotation @ reflection


def _select_text(
    episode: dict[str, Any], anchor_index: int
) -> tuple[str, dict[str, str], dict[str, tuple[int, int] | None]]:
    """Select the action active at the anchor and retain its half-open validity interval.

    VITRA's two hands have independent language windows.  Returning the intervals alongside the
    prompt prevents a secondary-hand label from silently supervising frames belonging to another
    action.  The original description is deterministic here; text rephrases remain available in
    the raw episode for a later language-augmentation pass.
    """
    side_text: dict[str, str] = {"left": "", "right": ""}
    side_interval: dict[str, tuple[int, int] | None] = {"left": None, "right": None}
    for side in ("left", "right"):
        for description, frame_range in episode.get("text", {}).get(side, []):
            start, end = int(frame_range[0]), int(frame_range[1])
            if start <= anchor_index < end:
                side_text[side] = str(description).strip()
                side_interval[side] = (start, end)
                break
    primary = str(episode.get("anno_type", "right")).lower()
    order = (primary, "left" if primary == "right" else "right") if primary in side_text else ("right", "left")
    clauses = [f"{side.capitalize()} hand: {side_text[side]}" for side in order if side_text[side]]
    return " ".join(clauses), side_text, side_interval


def enumerate_vitra_prompts(episode: dict[str, Any]) -> set[str]:
    """Return every prompt that anchor-dependent VITRA sampling can produce.

    The active language only changes at a half-open interval boundary, so one anchor from each
    boundary segment is sufficient and avoids visiting every frame in long videos.
    """
    length = int(len(episode["extrinsics"]))
    boundaries = {0, length}
    for side in ("left", "right"):
        for _, frame_range in episode.get("text", {}).get(side, []):
            start = min(max(int(frame_range[0]), 0), length)
            end = min(max(int(frame_range[1]), 0), length)
            boundaries.update((start, end))
    ordered = sorted(boundaries)
    prompts: set[str] = set()
    for start, end in zip(ordered, ordered[1:]):
        if start < end:
            prompts.add(_select_text(episode, start)[0])
    return prompts


def canonicalize_vitra_episode(
    episode: dict[str, Any],
    history_indices: Sequence[int],
    future_indices: Sequence[int],
    frame_times_seconds: Sequence[float],
    *,
    image_size: tuple[int, int] | None = None,
    calibration_size: tuple[float, float] | None = None,
    intrinsics_crop_xywh: tuple[float, float, float, float] | None = None,
    context_images: torch.Tensor | None = None,
    left_mano_policy: LeftManoPolicy = "as_stored",
    source_dataset: str | None = None,
    episode_id: str | None = None,
) -> dict[str, Any]:
    if left_mano_policy == "mirror_x":
        raise NotImplementedError(
            "Direct reflection into native MANO_LEFT rotations is not validated. Use "
            "left_mano_policy='as_stored' for VITRA's released right-canonical left pose."
        )
    history_indices = np.asarray(history_indices, dtype=np.int64)
    future_indices = np.asarray(future_indices, dtype=np.int64)
    if history_indices.ndim != 1 or future_indices.ndim != 1:
        raise ValueError("history_indices and future_indices must be one-dimensional")
    if len(history_indices) == 0 or len(future_indices) == 0:
        raise ValueError("Both history and future windows must be non-empty")
    anchor_index = int(history_indices[-1])
    query_indices = np.concatenate((history_indices, future_indices))

    extrinsics = _as_float_tensor(episode["extrinsics"])
    frame_times = np.asarray(frame_times_seconds, dtype=np.float64)
    length = int(extrinsics.shape[0])
    if frame_times.shape != (length,):
        raise ValueError(f"Expected {length} frame timestamps, got {tuple(frame_times.shape)}")
    if not np.isfinite(frame_times).all() or np.any(np.diff(frame_times) <= 0):
        raise ValueError("VITRA frame timestamps must be finite and strictly increasing")
    if query_indices.min() < 0 or query_indices.max() >= length:
        raise IndexError("Requested VITRA window lies outside the episode")
    indices = torch.as_tensor(query_indices, dtype=torch.long)
    anchor_world_to_camera = extrinsics[anchor_index]

    # Camera Ct -> anchor A. Released extrinsics map world -> camera.
    camera_pose = anchor_world_to_camera @ invert(extrinsics[indices])
    camera_pose9 = encode_pose9(camera_pose)

    wrist_pose9: dict[str, torch.Tensor] = {}
    mano_rot6: dict[str, torch.Tensor] = {}
    validity: dict[str, torch.Tensor] = {}
    fingertips: dict[str, torch.Tensor] = {}
    hand_joints_local: dict[str, torch.Tensor] = {}
    for side in ("left", "right"):
        side_data = episode[side]
        root_world = _make_transform(
            _as_float_tensor(side_data["global_orient_worldspace"])[indices],
            _as_float_tensor(side_data["transl_worldspace"])[indices],
        )
        wrist_pose9[side] = encode_pose9(anchor_world_to_camera @ root_world)

        local_rotations = _as_float_tensor(side_data["hand_pose"])[indices]
        if side == "left" and left_mano_policy == "mirror_x":
            local_rotations = _mirror_rotations_x(local_rotations)
        mano_rot6[side] = matrix_to_rotation_6d(local_rotations).reshape(len(indices), 90)

        side_validity = torch.as_tensor(
            np.asarray(side_data["kept_frames"])[query_indices].astype(bool)
        )
        validity[side] = side_validity

        joints_world = _as_float_tensor(side_data["joints_worldspace"])[indices]
        joints_anchor = transform_points(anchor_world_to_camera.expand(len(indices), -1, -1), joints_world)
        fingertips[side] = joints_anchor[:, FINGERTIP_INDICES]
        # Articulation targets should not contain global wrist or camera motion.  Expressing all
        # 21 joints in the instantaneous wrist-root frame gives the kinematic auxiliary head a
        # direct, metric target for finger shape while the canonical wrist stream handles SE(3).
        hand_joints_local[side] = transform_points(invert(root_world), joints_world)

    state = SCHEMA.pack(
        camera_pose9,
        wrist_pose9["left"],
        wrist_pose9["right"],
        mano_rot6["left"],
        mano_rot6["right"],
    )
    combined_text, side_text, side_interval = _select_text(episode, anchor_index)
    language_validity: dict[str, torch.Tensor] = {}
    for side in ("left", "right"):
        interval = side_interval[side]
        if interval is None:
            language_validity[side] = torch.zeros(len(indices), dtype=torch.bool)
        else:
            start, end = interval
            language_validity[side] = torch.as_tensor(
                (query_indices >= start) & (query_indices < end), dtype=torch.bool
            )

    stream_mask = torch.ones(len(indices), 5, dtype=torch.bool)
    stream_mask[:, 1] = validity["left"] & language_validity["left"]
    stream_mask[:, 2] = validity["right"] & language_validity["right"]
    stream_mask[:, 3] = validity["left"] & language_validity["left"]
    stream_mask[:, 4] = validity["right"] & language_validity["right"]
    if left_mano_policy == "mask":
        # Translation/fingertips remain available for adapter diagnostics, but root rotation and
        # articulation are not yet a validated side-specific physical representation. The v1
        # stream-level contract therefore masks the whole left wrist conservatively.
        stream_mask[:, 1] = False
        stream_mask[:, 3] = False

    # Subtract in float64 before casting. Hour-scale absolute PTS lose 16--33 ms intervals if
    # converted to float32 first.
    relative_time = torch.from_numpy(
        (frame_times[query_indices] - frame_times[anchor_index]).astype(np.float32)
    )
    history_length = len(history_indices)
    intrinsics_matrix = _as_float_tensor(episode["intrinsics"])
    # Normalize on the calibrated/cropped canvas first. A later isotropic or anisotropic resize
    # leaves normalized values unchanged. Dividing raw K by the resized frame size is wrong
    # (e.g. VITRA Ego4D K is calibrated at 1920x1440 while RGB is 720x540).
    if intrinsics_crop_xywh is not None:
        crop_x, crop_y, crop_width, crop_height = intrinsics_crop_xywh
        if crop_width <= 0 or crop_height <= 0:
            raise ValueError("intrinsics_crop_xywh must have positive width and height")
        normalized_fx = intrinsics_matrix[0, 0] / crop_width
        normalized_fy = intrinsics_matrix[1, 1] / crop_height
        normalized_cx = (intrinsics_matrix[0, 2] - crop_x) / crop_width
        normalized_cy = (intrinsics_matrix[1, 2] - crop_y) / crop_height
        intrinsics_source = "explicit_crop"
    else:
        if calibration_size is None:
            source_width = float(2.0 * intrinsics_matrix[0, 2])
            source_height = float(2.0 * intrinsics_matrix[1, 2])
            intrinsics_source = "inferred_centered_calibration_canvas"
        else:
            source_height, source_width = map(float, calibration_size)
            intrinsics_source = "explicit_calibration_canvas"
        if source_width <= 0 or source_height <= 0:
            raise ValueError("Calibration canvas must have positive dimensions")
        normalized_fx = intrinsics_matrix[0, 0] / source_width
        normalized_fy = intrinsics_matrix[1, 1] / source_height
        normalized_cx = intrinsics_matrix[0, 2] / source_width
        normalized_cy = intrinsics_matrix[1, 2] / source_height
    intrinsics = torch.tensor([normalized_fx, normalized_fy, normalized_cx, normalized_cy])
    intrinsics_normalized = True

    query_stream_mask = torch.ones(len(future_indices), 5, dtype=torch.bool)
    if left_mano_policy == "mask":
        query_stream_mask[:, 1] = False
        query_stream_mask[:, 3] = False
    output: dict[str, Any] = {
        "history_time": relative_time[:history_length],
        "history_query_mask": torch.ones(history_length, dtype=torch.bool),
        "history_state": state[:history_length],
        "history_stream_mask": stream_mask[:history_length],
        "future_time": relative_time[history_length:],
        "future_query_stream_mask": query_stream_mask,
        "future_state": state[history_length:],
        "future_stream_mask": stream_mask[history_length:],
        "history_fingertips": torch.stack(
            (fingertips["left"][:history_length], fingertips["right"][:history_length]),
            dim=1,
        ),
        "future_fingertips": torch.stack(
            (fingertips["left"][history_length:], fingertips["right"][history_length:]),
            dim=1,
        ),
        "history_hand_joints_local": torch.stack(
            (
                hand_joints_local["left"][:history_length],
                hand_joints_local["right"][:history_length],
            ),
            dim=1,
        ),
        "future_hand_joints_local": torch.stack(
            (
                hand_joints_local["left"][history_length:],
                hand_joints_local["right"][history_length:],
            ),
            dim=1,
        ),
        "text": combined_text,
        "intrinsics": intrinsics.float(),
        "metadata": {
            "video_name": str(episode.get("video_name", "")),
            "source_dataset": source_dataset,
            "episode_id": episode_id,
            "anchor_index": anchor_index,
            "video_decode_frames": np.asarray(episode["video_decode_frame"])[query_indices].tolist(),
            "left_mano_policy": left_mano_policy,
            "intrinsics_normalized": intrinsics_normalized,
            "intrinsics_source": intrinsics_source,
            "text_left": side_text["left"],
            "text_right": side_text["right"],
            "text_interval_left": side_interval["left"],
            "text_interval_right": side_interval["right"],
            "history_language_mask_left": language_validity["left"][:history_length].tolist(),
            "history_language_mask_right": language_validity["right"][:history_length].tolist(),
            "future_language_mask_left": language_validity["left"][history_length:].tolist(),
            "future_language_mask_right": language_validity["right"][history_length:].tolist(),
            "beta_left": np.asarray(episode["left"]["beta"]).tolist(),
            "beta_right": np.asarray(episode["right"]["beta"]).tolist(),
        },
    }
    if context_images is not None:
        if context_images.shape[0] != history_length:
            raise ValueError("context_images must align with history_indices")
        output["context_images"] = context_images
    return output
