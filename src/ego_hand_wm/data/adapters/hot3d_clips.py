"""Official HOT3D-Clips Aria tar files to the canonical geometry contract.

The public HOT3D-Clips training archives contain metric world poses for the
Aria RGB camera and for both UmeTrack wrists.  This adapter deliberately does
not reinterpret the packaged 15-dimensional MANO parameter vector as fifteen
axis-angle rotations: it is a hand-model parameterization and requires the
official hand-tracking toolkit/model to decode correctly.  Consequently the
canonical MANO streams remain masked. The controlled H6/K16 benchmark
supervises wrist XYZ by default, matching the adapted H2O and EgoPAT3D
targets; full wrist SE(3) remains available as an explicit ablation.
"""

from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np
import torch
import torch.nn.functional as functional
from PIL import Image

from ego_hand_wm.contracts.schema import GEOMETRY_DIM, SCHEMA
from ego_hand_wm.geometry.se3 import encode_pose9, invert

Hot3DAnchor = Literal["last_observed", "first_observed"]
Hot3DWristTarget = Literal["xyz", "se3"]


def quaternion_wxyz_to_matrix(quaternion: Sequence[float]) -> torch.Tensor:
    """Convert a HOT3D ``[w,x,y,z]`` quaternion to a rotation matrix."""

    value = torch.as_tensor(quaternion, dtype=torch.float64)
    if value.shape != (4,):
        raise ValueError(f"Expected quaternion [w,x,y,z], got {tuple(value.shape)}")
    norm = torch.linalg.vector_norm(value)
    if not torch.isfinite(norm) or norm <= 0:
        raise ValueError("HOT3D quaternion must be finite and nonzero")
    w, x, y, z = value / norm
    return torch.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        )
    ).reshape(3, 3)


def se3_from_hot3d(value: dict[str, Any]) -> torch.Tensor:
    """Decode a HOT3D quaternion/translation object as ``T_parent_from_child``."""

    translation = torch.as_tensor(value["translation_xyz"], dtype=torch.float64)
    if translation.shape != (3,) or not torch.isfinite(translation).all():
        raise ValueError("HOT3D translation_xyz must contain three finite values")
    transform = torch.eye(4, dtype=torch.float64)
    transform[:3, :3] = quaternion_wxyz_to_matrix(value["quaternion_wxyz"])
    transform[:3, 3] = translation
    return transform


def _read_json(archive: tarfile.TarFile, member_name: str) -> dict[str, Any]:
    member = archive.extractfile(member_name)
    if member is None:
        raise FileNotFoundError(f"Missing {member_name} in {archive.name}")
    value = json.load(member)
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object in {member_name}")
    return value


def _hand_world_transform(hands: dict[str, Any], side: str) -> torch.Tensor | None:
    hand = hands.get(side)
    if not isinstance(hand, dict):
        return None
    umetrack = hand.get("umetrack_pose")
    if not isinstance(umetrack, dict):
        return None
    transform = umetrack.get("T_world_from_wrist")
    if not isinstance(transform, dict):
        return None
    return se3_from_hot3d(transform)


def hot3d_clip_hand_validity(
    tar_path: str | Path, *, num_frames: int = 150
) -> dict[str, tuple[bool, ...]]:
    """Return per-frame public hand-label validity without decoding images."""

    result = {"left": [], "right": []}
    with tarfile.open(tar_path, "r") as archive:
        names = set(archive.getnames())
        for frame_index in range(num_frames):
            member_name = f"{frame_index:06d}.hands.json"
            if member_name not in names:
                for side in result:
                    result[side].append(False)
                continue
            hands = _read_json(archive, member_name)
            for side in result:
                result[side].append(_hand_world_transform(hands, side) is not None)
    return {side: tuple(values) for side, values in result.items()}


def _decode_rgb(
    archive: tarfile.TarFile,
    frame_indices: Sequence[int],
    *,
    stream_id: str,
    image_size: int | None,
    rotate_clockwise: bool,
) -> torch.Tensor:
    images: list[torch.Tensor] = []
    for frame_index in frame_indices:
        name = f"{frame_index:06d}.image_{stream_id}.jpg"
        member = archive.extractfile(name)
        if member is None:
            raise FileNotFoundError(f"Missing {name} in {archive.name}")
        image = Image.open(io.BytesIO(member.read())).convert("RGB")
        if rotate_clockwise:
            image = image.transpose(Image.Transpose.ROTATE_270)
        array = np.array(image, dtype=np.uint8, copy=True)
        images.append(torch.from_numpy(array).permute(2, 0, 1))
    result = torch.stack(images).float().div_(255.0)
    if image_size is not None:
        result = functional.interpolate(
            result,
            size=(image_size, image_size),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
    return result


def _normalized_intrinsics(
    calibration: dict[str, Any], *, rotate_clockwise: bool
) -> torch.Tensor:
    parameters = calibration["projection_params"]
    if len(parameters) < 3:
        raise ValueError("HOT3D FISHEYE624 calibration lacks f,cx,cy")
    focal, center_x, center_y = map(float, parameters[:3])
    width = float(calibration["image_width"])
    height = float(calibration["image_height"])
    if rotate_clockwise:
        # x' = H - 1 - y, y' = x for the RGB rotation used by HOT3D examples.
        center_x, center_y = height - 1.0 - center_y, center_x
        width, height = height, width
    return torch.tensor(
        (focal / width, focal / height, center_x / width, center_y / height),
        dtype=torch.float32,
    )


def canonicalize_hot3d_clip_window(
    tar_path: str | Path,
    history_indices: Sequence[int],
    future_indices: Sequence[int],
    history_time_seconds: Sequence[float],
    future_time_seconds: Sequence[float],
    *,
    tracked_hand: Literal["left", "right"],
    camera_stream: str = "214-1",
    anchor: Hot3DAnchor = "last_observed",
    wrist_target: Hot3DWristTarget = "xyz",
    decode_rgb: bool = False,
    image_size: int | None = None,
    rotate_clockwise: bool = True,
) -> dict[str, Any]:
    """Read one official clip window and express it in one anchored frame.

    The default spatial anchor is the last observed Aria RGB camera, matching
    VITRA and the H2O/EgoPAT3D adapters.  ``first_observed`` is available for
    papers that define the prediction canvas at the first clip frame.  Time
    zero remains the last observation in both cases.
    """

    if tracked_hand not in {"left", "right"}:
        raise ValueError("tracked_hand must be 'left' or 'right'")
    if anchor not in {"last_observed", "first_observed"}:
        raise ValueError("anchor must be 'last_observed' or 'first_observed'")
    if wrist_target not in {"xyz", "se3"}:
        raise ValueError("wrist_target must be 'xyz' or 'se3'")
    history_indices = tuple(int(value) for value in history_indices)
    future_indices = tuple(int(value) for value in future_indices)
    if not history_indices or not future_indices:
        raise ValueError("HOT3D forecasting requires nonempty history and future")
    selected = history_indices + future_indices
    if len(set(selected)) != len(selected) or min(selected) < 0:
        raise ValueError("HOT3D frame indices must be unique and nonnegative")

    camera_world: list[torch.Tensor] = []
    wrist_world = {"left": [], "right": []}
    wrist_valid = {"left": [], "right": []}
    actual_timestamps_ns: list[int] = []
    calibration: dict[str, Any] | None = None
    with tarfile.open(tar_path, "r") as archive:
        for frame_index in selected:
            frame_key = f"{frame_index:06d}"
            cameras = _read_json(archive, f"{frame_key}.cameras.json")
            if camera_stream not in cameras:
                raise KeyError(f"Camera stream {camera_stream!r} missing at frame {frame_key}")
            camera = cameras[camera_stream]
            camera_world.append(se3_from_hot3d(camera["T_world_from_camera"]))
            calibration = camera["calibration"]
            info = _read_json(archive, f"{frame_key}.info.json")
            timestamp_map = info.get("image_timestamps_ns", {})
            timestamp = timestamp_map.get(camera_stream, info["ref_timestamp_ns"])
            actual_timestamps_ns.append(int(timestamp))
            hands = _read_json(archive, f"{frame_key}.hands.json")
            for side in ("left", "right"):
                transform = _hand_world_transform(hands, side)
                wrist_valid[side].append(transform is not None)
                wrist_world[side].append(
                    transform if transform is not None else torch.eye(4, dtype=torch.float64)
                )
        context_images = (
            _decode_rgb(
                archive,
                history_indices,
                stream_id=camera_stream,
                image_size=image_size,
                rotate_clockwise=rotate_clockwise,
            )
            if decode_rgb
            else None
        )

    if calibration is None:
        raise RuntimeError(f"No camera calibration decoded from {tar_path}")
    camera_world_tensor = torch.stack(camera_world)
    spatial_anchor_position = len(history_indices) - 1 if anchor == "last_observed" else 0
    anchor_from_world = invert(camera_world_tensor[spatial_anchor_position])
    camera_anchor = anchor_from_world.unsqueeze(0) @ camera_world_tensor
    wrist_anchor = {
        side: anchor_from_world.unsqueeze(0) @ torch.stack(wrist_world[side])
        for side in ("left", "right")
    }

    count = len(selected)
    empty_mano = torch.zeros(count, 90, dtype=torch.float32)
    state = SCHEMA.pack(
        encode_pose9(camera_anchor).float(),
        encode_pose9(wrist_anchor["left"]).float(),
        encode_pose9(wrist_anchor["right"]).float(),
        empty_mano,
        empty_mano.clone(),
    )
    stream_mask = torch.zeros(count, 5, dtype=torch.bool)
    stream_mask[:, 0] = True
    stream_mask[:, 1] = torch.tensor(wrist_valid["left"], dtype=torch.bool)
    stream_mask[:, 2] = torch.tensor(wrist_valid["right"], dtype=torch.bool)
    state_mask = SCHEMA.expand_stream_mask(stream_mask)
    if wrist_target == "xyz":
        # HOT3D exposes wrist orientation, but the controlled comparison is a
        # point-trajectory benchmark. Mask rotation exactly as the canonical
        # H2O/EgoPAT3D adapter does, while retaining it in ``state`` for an
        # explicitly requested SE(3) ablation.
        state_mask[:, SCHEMA.left_wrist.start + 3 : SCHEMA.left_wrist.stop] = False
        state_mask[:, SCHEMA.right_wrist.start + 3 : SCHEMA.right_wrist.stop] = False

    history_count = len(history_indices)
    future_count = len(future_indices)
    query_stream_mask = torch.zeros(future_count, 5, dtype=torch.bool)
    query_stream_mask[:, 0] = True
    tracked_stream = 1 if tracked_hand == "left" else 2
    query_stream_mask[:, tracked_stream] = True
    future_stream_mask = stream_mask[history_count:] & query_stream_mask
    future_state_mask = (
        state_mask[history_count:]
        & SCHEMA.expand_stream_mask(query_stream_mask)
    )

    history_time = torch.as_tensor(history_time_seconds, dtype=torch.float32)
    future_time = torch.as_tensor(future_time_seconds, dtype=torch.float32)
    if history_time.shape != (history_count,) or future_time.shape != (future_count,):
        raise ValueError("Manifest time arrays must align with HOT3D frame indices")
    actual_time = (
        np.asarray(actual_timestamps_ns, dtype=np.float64)
        - float(actual_timestamps_ns[history_count - 1])
    ) / 1e9

    sample: dict[str, Any] = {
        "history_time": history_time,
        "history_query_mask": torch.ones(history_count, dtype=torch.bool),
        "history_state": state[:history_count],
        "history_stream_mask": stream_mask[:history_count],
        "history_state_mask": state_mask[:history_count],
        "future_time": future_time,
        "future_query_stream_mask": query_stream_mask,
        "future_state": state[history_count:],
        "future_stream_mask": future_stream_mask,
        "future_state_mask": future_state_mask,
        "text": "",
        "intrinsics": _normalized_intrinsics(
            calibration, rotate_clockwise=rotate_clockwise
        ),
        "metadata": {
            "tracked_hand": tracked_hand,
            "tracked_wrist_stream": tracked_stream,
            "camera_stream": camera_stream,
            "anchor": f"{anchor}_camera",
            "canonical_camera": "T_A_from_Ct",
            "wrist_pose_source": "umetrack_pose.T_world_from_wrist",
            "wrist_target": wrist_target,
            "mano_streams": "masked_requires_official_hand_model_decode",
            "actual_time_seconds": actual_time.tolist(),
            "rgb_rotated_clockwise": rotate_clockwise,
            "supervised_geometry": (
                "camera_se3_and_tracked_wrist_xyz"
                if wrist_target == "xyz"
                else "camera_se3_and_tracked_wrist_se3"
            ),
        },
    }
    if state.shape != (count, GEOMETRY_DIM):
        raise AssertionError("HOT3D adapter emitted an invalid canonical state")
    if context_images is not None:
        sample["context_images"] = context_images
    return sample
