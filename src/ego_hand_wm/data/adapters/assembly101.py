"""Assembly101 e4 geometry conversion for semantic anticipation.

Assembly101 releases action boundaries on a 30 fps extraction, while the videos and pose
archive are 60 Hz.  The pose archive stores camera and hand transforms in millimetres in a
shared world frame.  This adapter keeps those dataset-specific details outside the anticipation
model and emits camera/wrist SE(3) in the final observed e4 camera frame, matching the VITRA
``A=C0`` convention used elsewhere in this repository.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from ego_hand_wm.geometry.se3 import encode_pose9, invert, transform_points


ANNOTATION_FPS = 30
RAW_FPS = 60
RAW_FRAMES_PER_ANNOTATION_FRAME = RAW_FPS // ANNOTATION_FPS
E4_VIDEO_STEMS = frozenset(
    {
        "HMC_21179183_mono10bit",
        "HMC_84358933_mono10bit",
    }
)

WristReference = Literal["camera_anchor", "last_observed_wrist"]
WRIST_REFERENCES = frozenset({"camera_anchor", "last_observed_wrist"})


def is_e4_video(video: str | Path) -> bool:
    """Return whether an annotation video path is the physical e4 headset view."""

    return Path(video).stem in E4_VIDEO_STEMS


def e4_pose_camera_key(video: str | Path) -> str:
    """Map an e4 MP4 stem to the camera key used inside ``AssemblyPoses.zip``."""

    stem = Path(video).stem
    if stem not in E4_VIDEO_STEMS:
        raise ValueError(f"Not an Assembly101 e4 camera: {stem}")
    match = re.fullmatch(r"HMC_(\d+)_mono10bit", stem)
    if match is None:  # protected by E4_VIDEO_STEMS; retained as a schema guard
        raise ValueError(f"Unexpected Assembly101 ego-camera stem: {stem}")
    return f"{match.group(1)}:mono10bit"


def annotation_to_raw_frame(frame: int) -> int:
    """Convert a released 30 fps action index to its aligned raw/pose 60 Hz index."""

    if frame < 0:
        raise ValueError("Assembly101 frame indices must be non-negative")
    return int(frame) * RAW_FRAMES_PER_ANNOTATION_FRAME


def _as_float_tensor(value: Any) -> torch.Tensor:
    return torch.as_tensor(np.asarray(value), dtype=torch.float32)


def canonicalize_assembly101_geometry(
    camera_world_from_camera: np.ndarray | torch.Tensor,
    wrist_world_from_hand: np.ndarray | torch.Tensor,
    wrist_confidence: np.ndarray | torch.Tensor,
    *,
    confidence_threshold: float = 0.25,
) -> dict[str, torch.Tensor]:
    """Express an observed e4 geometry window in its final camera frame.

    Args:
        camera_world_from_camera: ``[T,4,4]`` e4 camera-to-world transforms in mm.
        wrist_world_from_hand: ``[T,2,4,4]`` tracker root transforms in mm.  The two stable
            tracker slots are intentionally named hand-0 and hand-1: the raw release does not
            document a handedness mapping for these JSON keys.
        wrist_confidence: ``[T,2]`` released tracker confidence values.
        confidence_threshold: confidence at or above this value is a valid wrist token.

    Returns:
        Camera pose-9 ``[T,9]``, two wrist pose-9 streams ``[T,2,9]``, confidence and validity.
        All translations are converted from millimetres to metres.  The final camera pose is
        identity by construction.
    """

    camera = _as_float_tensor(camera_world_from_camera)
    wrist = _as_float_tensor(wrist_world_from_hand)
    confidence = _as_float_tensor(wrist_confidence)
    if camera.ndim != 3 or camera.shape[-2:] != (4, 4) or camera.shape[0] == 0:
        raise ValueError("camera_world_from_camera must have non-empty shape [T,4,4]")
    expected_wrist = (camera.shape[0], 2, 4, 4)
    if tuple(wrist.shape) != expected_wrist:
        raise ValueError(f"Expected wrist transforms {expected_wrist}, got {tuple(wrist.shape)}")
    if tuple(confidence.shape) != (camera.shape[0], 2):
        raise ValueError("wrist_confidence must have shape [T,2]")
    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold must lie in [0,1]")
    if not torch.isfinite(camera).all() or not torch.isfinite(wrist).all():
        raise ValueError("Assembly101 transforms must be finite")

    anchor_from_world = invert(camera[-1])
    camera_anchor_from_camera = anchor_from_world.unsqueeze(0) @ camera
    wrist_anchor_from_hand = anchor_from_world.view(1, 1, 4, 4) @ wrist

    camera_pose = encode_pose9(camera_anchor_from_camera)
    wrist_pose = encode_pose9(wrist_anchor_from_hand)
    camera_pose[..., :3] /= 1000.0
    wrist_pose[..., :3] /= 1000.0

    confidence = confidence.clamp(0.0, 1.0)
    wrist_valid = confidence >= float(confidence_threshold)
    # Invalid tracker frames frequently contain an identity placeholder in the release.  Zeroing
    # prevents that placeholder from becoming a large, plausible anchor-relative hand pose.
    wrist_pose = torch.where(wrist_valid[..., None], wrist_pose, torch.zeros_like(wrist_pose))
    return {
        "camera_pose": camera_pose,
        "wrist_pose": wrist_pose,
        "wrist_confidence": confidence,
        "wrist_valid": wrist_valid,
    }


def canonicalize_assembly101_oracle_geometry(
    camera_world_from_camera: np.ndarray | torch.Tensor,
    wrist_world_from_hand: np.ndarray | torch.Tensor,
    landmarks_world: np.ndarray | torch.Tensor,
    wrist_confidence: np.ndarray | torch.Tensor,
    *,
    anchor_index: int,
    confidence_threshold: float = 0.25,
    wrist_reference: WristReference = "camera_anchor",
) -> dict[str, torch.Tensor]:
    """Canonicalize a joint past/future window around the legal observation anchor.

    Unlike :func:`canonicalize_assembly101_geometry`, the input may extend into the oracle
    future. Consequently, the anchor must be supplied explicitly and must point to the last
    observed frame rather than the final frame of the combined tensor.

    ``camera_anchor`` preserves the completed experiment convention: wrist roots are positions
    in the last observed camera frame. ``last_observed_wrist`` instead expresses each hand root
    relative to that same hand at the legal observation cutoff. It therefore exposes cumulative
    past/future wrist motion directly and makes the wrist pose identity at ``anchor_index``.

    The released 21-point landmarks are always returned in their instantaneous wrist coordinate
    frame. This removes both wrist translation and wrist orientation, giving the `handpose-only`
    ablation an articulation signal that is distinct from the wrist-trajectory ablation.
    """

    camera = _as_float_tensor(camera_world_from_camera)
    wrist = _as_float_tensor(wrist_world_from_hand)
    landmarks = _as_float_tensor(landmarks_world)
    confidence = _as_float_tensor(wrist_confidence)
    if camera.ndim != 3 or camera.shape[-2:] != (4, 4) or camera.shape[0] == 0:
        raise ValueError("camera_world_from_camera must have non-empty shape [T,4,4]")
    steps = camera.shape[0]
    if tuple(wrist.shape) != (steps, 2, 4, 4):
        raise ValueError("wrist_world_from_hand must have shape [T,2,4,4]")
    if tuple(landmarks.shape) != (steps, 2, 21, 3):
        raise ValueError("landmarks_world must have shape [T,2,21,3]")
    if tuple(confidence.shape) != (steps, 2):
        raise ValueError("wrist_confidence must have shape [T,2]")
    if not 0 <= anchor_index < steps:
        raise IndexError(f"anchor_index {anchor_index} is outside a {steps}-frame window")
    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold must lie in [0,1]")
    if wrist_reference not in WRIST_REFERENCES:
        raise ValueError(
            f"Unsupported wrist_reference={wrist_reference!r}; expected one of "
            f"{sorted(WRIST_REFERENCES)}"
        )

    transform_finite = torch.isfinite(camera).all(dim=(-2, -1)) & torch.isfinite(wrist).all(
        dim=(-2, -1)
    ).all(dim=-1)
    if not transform_finite.all():
        raise ValueError("Assembly101 camera/wrist transforms must be finite")
    landmark_finite = torch.isfinite(landmarks).all(dim=(-2, -1))

    anchor_from_world = invert(camera[anchor_index])
    camera_anchor_from_camera = anchor_from_world.unsqueeze(0) @ camera
    if wrist_reference == "camera_anchor":
        wrist_canonical_from_hand = anchor_from_world.view(1, 1, 4, 4) @ wrist
    else:
        # Each hand has its own legal observed anchor. For every past and future timestamp this
        # yields the cumulative SE(3) motion from t=0, with identity at anchor_index. No future
        # pose participates in constructing the reference.
        wrist_anchor_from_world = invert(wrist[anchor_index])
        wrist_canonical_from_hand = wrist_anchor_from_world.unsqueeze(0) @ wrist
    hand_from_world = invert(wrist)
    hand_pose = transform_points(hand_from_world, landmarks)

    camera_pose = encode_pose9(camera_anchor_from_camera)
    wrist_pose = encode_pose9(wrist_canonical_from_hand)
    camera_pose[..., :3] /= 1000.0
    wrist_pose[..., :3] /= 1000.0
    hand_pose /= 1000.0

    confidence = confidence.clamp(0.0, 1.0)
    tracked_valid = confidence >= float(confidence_threshold)
    hand_pose_valid = tracked_valid & landmark_finite
    wrist_valid = tracked_valid
    if wrist_reference == "last_observed_wrist":
        # A motion trajectory has no well-defined reference if the released tracker is invalid at
        # t=0. Mask that hand for the complete window instead of silently shifting its anchor.
        wrist_valid = wrist_valid & tracked_valid[anchor_index].unsqueeze(0)
    wrist_pose = torch.where(wrist_valid[..., None], wrist_pose, torch.zeros_like(wrist_pose))
    hand_pose = torch.where(
        hand_pose_valid[..., None, None], hand_pose, torch.zeros_like(hand_pose)
    )
    return {
        "camera_pose": camera_pose,
        "wrist_pose": wrist_pose,
        "hand_pose": hand_pose,
        "wrist_confidence": confidence,
        "wrist_valid": wrist_valid,
        "hand_pose_valid": hand_pose_valid,
    }
