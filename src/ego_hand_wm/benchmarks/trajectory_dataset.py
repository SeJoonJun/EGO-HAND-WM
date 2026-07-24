"""Shared H2O/EgoPAT3D manifest dataset for adapted trajectory baselines."""

from __future__ import annotations

import json
import pickle
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as functional
from torch.utils.data import Dataset

from ego_hand_wm.contracts.schema import GEOMETRY_DIM, SCHEMA
from ego_hand_wm.data.trajectory_features import TrajectoryVisualFeatureStore
from ego_hand_wm.geometry.se3 import encode_pose9


@lru_cache(maxsize=32)
def _read_pickle(path: str) -> dict[str, Any]:
    with Path(path).open("rb") as handle:
        value = pickle.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected dictionary in {path}")
    return value


@lru_cache(maxsize=32)
def _read_numpy(path: str) -> np.ndarray:
    """Cache clip-level arrays shared by many overlapping trajectory windows."""
    value = np.load(path, allow_pickle=False)
    if not isinstance(value, np.ndarray):
        raise TypeError(f"Expected NumPy array in {path}")
    return value


def _transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    homogeneous = np.concatenate(
        (points, np.ones((len(points), 1), dtype=points.dtype)), axis=-1
    )
    return (transform @ homogeneous.T).T[:, :3]


def _h2o_anchor_window(
    record: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    payload = _read_pickle(record["trajectory_path"])
    hand_key = f"{record['hand']}_hand"
    segment = payload[hand_key][int(record["trajectory_segment_index"])]
    start = int(record["trajectory_window_start"])
    total = int(record["history_steps"]) + int(record["future_steps"])
    end = start + total
    world_points = np.asarray(segment["traj3d"], dtype=np.float64)[start:end]
    cam2world = np.asarray(segment["cam2world"], dtype=np.float64)[start:end]
    if len(world_points) != total or len(cam2world) != total:
        raise ValueError(f"Incomplete H2O window: {record['sample_id']}")
    anchor_local = int(record["history_steps"]) - 1
    world_to_anchor = np.linalg.inv(cam2world[anchor_local])
    camera_to_anchor = world_to_anchor[None] @ cam2world
    intrinsics = payload.get(
        "intrinsics",
        {
            "fx": 636.6593017578125,
            "fy": 636.251953125,
            "cx": 635.283881879317,
            "cy": 366.8740353496978,
            "width": 1280.0,
            "height": 720.0,
        },
    )
    normalized_intrinsics = np.asarray(
        (
            float(intrinsics["fx"]) / float(intrinsics["width"]),
            float(intrinsics["fy"]) / float(intrinsics["height"]),
            float(intrinsics["cx"]) / float(intrinsics["width"]),
            float(intrinsics["cy"]) / float(intrinsics["height"]),
        ),
        dtype=np.float32,
    )
    return (
        _transform_points(world_to_anchor, world_points).astype(np.float32),
        camera_to_anchor.astype(np.float32),
        normalized_intrinsics,
    )


def _h2o_anchor_trajectory(record: dict[str, Any]) -> np.ndarray:
    return _h2o_anchor_window(record)[0]


def _product(transforms: np.ndarray) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    for transform in transforms:
        result = result @ transform
    return result


def _egopat3d_anchor_window(
    record: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    payload = _read_pickle(record["trajectory_path"])
    start = int(record["trajectory_window_start"])
    history = int(record["history_steps"])
    total = history + int(record["future_steps"])
    end = start + total
    points = np.asarray(payload["traj3d"], dtype=np.float64)
    odometry = np.asarray(_read_numpy(record["odometry_path"]), dtype=np.float64)
    if end > len(points) or end > len(odometry):
        raise ValueError(f"Incomplete EgoPAT3D window: {record['sample_id']}")
    anchor = start + history - 1
    anchored: list[np.ndarray] = []
    camera_to_anchor: list[np.ndarray] = []
    for frame_index in range(start, end):
        if frame_index == anchor:
            frame_to_anchor = np.eye(4)
        elif frame_index > anchor:
            frame_to_anchor = _product(odometry[anchor + 1 : frame_index + 1])
        else:
            anchor_to_frame = _product(odometry[frame_index + 1 : anchor + 1])
            frame_to_anchor = np.linalg.inv(anchor_to_frame)
        point = np.concatenate((points[frame_index], np.ones(1)), axis=0)
        anchored.append(frame_to_anchor @ point)
        camera_to_anchor.append(frame_to_anchor)
    # Official EgoPAT3D calibration.  The released clips preserve the original projection
    # coordinates even when their RGB is resized for a backbone, so normalize by the native
    # 3840x2160 calibration canvas rather than by the model input crop.
    normalized_intrinsics = np.asarray(
        (1808.2 / 3840.0, 1807.95 / 2160.0, 1942.29 / 3840.0, 1123.82 / 2160.0),
        dtype=np.float32,
    )
    return (
        np.asarray(anchored, dtype=np.float32)[:, :3],
        np.asarray(camera_to_anchor, dtype=np.float32),
        normalized_intrinsics,
    )


def _egopat3d_anchor_trajectory(record: dict[str, Any]) -> np.ndarray:
    return _egopat3d_anchor_window(record)[0]


def _decode_video_frames(
    path: str, indices: list[int], *, image_size: int | None = None
) -> torch.Tensor:
    try:
        import av
    except ImportError as error:
        raise RuntimeError("PyAV is required when decode_rgb=True") from error

    requested = set(indices)
    decoded: dict[int, torch.Tensor] = {}
    with av.open(path) as container:
        for frame_index, frame in enumerate(container.decode(video=0)):
            if frame_index in requested:
                array = frame.to_ndarray(format="rgb24")
                decoded[frame_index] = torch.from_numpy(array).permute(2, 0, 1)
            if frame_index >= indices[-1]:
                break
    missing = [index for index in indices if index not in decoded]
    if missing:
        raise IndexError(f"Video {path} is missing requested frames {missing[:5]}")
    images = torch.stack([decoded[index] for index in indices]).float().div_(255.0)
    if image_size is not None:
        images = functional.interpolate(
            images,
            size=(image_size, image_size),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
    return images


class TrajectoryWindowDataset(Dataset):
    """Read the common JSONL contract without changing either source dataset."""

    def __init__(
        self,
        manifest: str | Path,
        *,
        decode_rgb: bool = False,
        image_size: int | None = None,
    ) -> None:
        self.manifest = Path(manifest)
        self.decode_rgb = decode_rgb
        self.image_size = image_size
        self.records = [
            json.loads(line)
            for line in self.manifest.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        for record in self.records:
            if record.get("protocol") != "h6_k16_30hz":
                raise ValueError(f"Unsupported protocol in {record.get('sample_id')}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        if record["dataset"] == "h2o":
            trajectory = _h2o_anchor_trajectory(record)
        elif record["dataset"] == "egopat3d":
            trajectory = _egopat3d_anchor_trajectory(record)
        else:
            raise ValueError(f"Unknown trajectory dataset: {record['dataset']}")
        history_steps = int(record["history_steps"])
        sample: dict[str, Any] = {
            "history_xyz_anchor": torch.from_numpy(trajectory[:history_steps]),
            "future_xyz_anchor": torch.from_numpy(trajectory[history_steps:]),
            "history_time": torch.tensor(record["history_time_seconds"], dtype=torch.float32),
            "future_time": torch.tensor(record["future_time_seconds"], dtype=torch.float32),
            "history_frame_indices": torch.tensor(record["history_indices"], dtype=torch.long),
            "future_frame_indices": torch.tensor(record["future_indices"], dtype=torch.long),
            "video_path": record["video_path"],
            "metadata": record,
        }
        if self.decode_rgb:
            sample["context_images"] = _decode_video_frames(
                record["video_path"],
                record["history_indices"],
                image_size=self.image_size,
            )
        return sample


class CanonicalTrajectoryDataset(Dataset):
    """Map H2O/EgoPAT3D H6/K16 windows into the VITRA canonical model contract.

    Both datasets provide one 3D hand point, not a full MANO wrist frame.  We place that point
    in the left/right wrist translation slots, safe-fill its rotation with identity, and expose
    a coordinate mask that supervises XYZ only.  Camera SE(3) is fully observed and represented
    as ``T_A_from_Ct``, exactly matching the VITRA adapter.
    """

    provides_future_visual = False

    def __init__(self, config: dict[str, Any]) -> None:
        split = str(config.get("split", "train"))
        manifests = config.get("manifests")
        if manifests is not None:
            try:
                manifest = manifests[split]
            except KeyError as error:
                raise KeyError(f"No trajectory manifest configured for split {split!r}") from error
        else:
            manifest = config["manifest"]
        self.base = TrajectoryWindowDataset(
            manifest,
            decode_rgb=bool(config.get("decode_rgb", False)),
            image_size=(
                int(config["image_size"])
                if config.get("image_size") is not None
                else None
            ),
        )
        datasets = {str(record["dataset"]) for record in self.base.records}
        splits = {str(record["split"]) for record in self.base.records}
        if len(datasets) != 1 or len(splits) != 1:
            raise ValueError("Each canonical trajectory manifest must contain one dataset/split")
        self.dataset_name = next(iter(datasets))
        self.split = next(iter(splits))
        split_aliases = {"validation": "val", **dict(config.get("split_aliases", {}))}
        expected_manifest_split = str(split_aliases.get(split, split))
        if self.split != expected_manifest_split:
            raise ValueError(
                f"Manifest split {self.split!r} does not match requested {split!r} "
                f"(expected {expected_manifest_split!r})"
            )
        feature_root = config.get("visual_feature_root")
        self.visual_store = (
            TrajectoryVisualFeatureStore(
                feature_root,
                dataset=self.dataset_name,
                split=self.split,
                sequence="history",
                output_dtype=np.dtype(config.get("visual_feature_dtype", "float32")),
            )
            if feature_root is not None
            else None
        )
        if self.visual_store is not None and self.base.decode_rgb:
            raise ValueError("Choose cached DINO.txt features or online RGB, not both")
        self.provides_context_visual = self.visual_store is not None or self.base.decode_rgb
        future_feature_root = config.get("future_visual_feature_root")
        future_splits = {
            str(value) for value in config.get("future_visual_splits", (self.split,))
        }
        requested_split = str(config.get("split", "train"))
        attach_future = self.split in future_splits or requested_split in future_splits
        self.future_visual_store = (
            TrajectoryVisualFeatureStore(
                future_feature_root,
                dataset=self.dataset_name,
                split=self.split,
                sequence="future",
                output_dtype=np.dtype(config.get("visual_feature_dtype", "float32")),
            )
            if future_feature_root is not None and attach_future
            else None
        )
        self.provides_future_visual = self.future_visual_store is not None
        self.text_feature_dim = int(config.get("missing_text_feature_dim", 0))
        self.egopat_hand = str(config.get("egopat_hand", "right"))
        if self.egopat_hand not in {"left", "right"}:
            raise ValueError("data.egopat_hand must be 'left' or 'right'")

    def __len__(self) -> int:
        return len(self.base)

    @staticmethod
    def _pose_state(
        trajectory: np.ndarray,
        camera_to_anchor: np.ndarray,
        *,
        hand: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        length = len(trajectory)
        camera = encode_pose9(torch.from_numpy(camera_to_anchor))
        wrist_transform = torch.eye(4, dtype=torch.float32).repeat(length, 1, 1)
        wrist_transform[:, :3, 3] = torch.from_numpy(trajectory)
        wrist = encode_pose9(wrist_transform)
        empty_wrist = torch.zeros(length, 9, dtype=torch.float32)
        empty_mano = torch.zeros(length, 90, dtype=torch.float32)
        state = SCHEMA.pack(
            camera,
            wrist if hand == "left" else empty_wrist,
            wrist if hand == "right" else empty_wrist,
            empty_mano,
            empty_mano,
        )
        stream_mask = torch.zeros(length, 5, dtype=torch.bool)
        stream_mask[:, 0] = True
        hand_stream = 1 if hand == "left" else 2
        stream_mask[:, hand_stream] = True
        component_mask = torch.zeros(length, GEOMETRY_DIM, dtype=torch.bool)
        component_mask[:, SCHEMA.camera] = True
        wrist_slice = SCHEMA.left_wrist if hand == "left" else SCHEMA.right_wrist
        component_mask[:, wrist_slice.start : wrist_slice.start + 3] = True
        return state, stream_mask, component_mask

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.base.records[index]
        if record["dataset"] == "h2o":
            trajectory, camera_to_anchor, intrinsics = _h2o_anchor_window(record)
            hand = str(record["hand"])
        elif record["dataset"] == "egopat3d":
            trajectory, camera_to_anchor, intrinsics = _egopat3d_anchor_window(record)
            hand = self.egopat_hand
        else:  # guarded by TrajectoryWindowDataset, retained for direct diagnostics
            raise ValueError(f"Unknown trajectory dataset: {record['dataset']}")
        history_steps = int(record["history_steps"])
        state, stream_mask, component_mask = self._pose_state(
            trajectory, camera_to_anchor, hand=hand
        )
        query_stream_mask = stream_mask[history_steps:].clone()
        sample: dict[str, Any] = {
            "history_time": torch.tensor(record["history_time_seconds"], dtype=torch.float32),
            "history_query_mask": torch.ones(history_steps, dtype=torch.bool),
            "history_state": state[:history_steps],
            "history_stream_mask": stream_mask[:history_steps],
            "history_state_mask": component_mask[:history_steps],
            "future_time": torch.tensor(record["future_time_seconds"], dtype=torch.float32),
            "future_query_stream_mask": query_stream_mask,
            "future_state": state[history_steps:],
            "future_stream_mask": stream_mask[history_steps:],
            "future_state_mask": component_mask[history_steps:],
            "text": "",
            "intrinsics": torch.from_numpy(intrinsics),
            "metadata": {
                **record,
                "source_dataset": str(record["dataset"]),
                "tracked_hand": hand,
                "tracked_wrist_stream": 1 if hand == "left" else 2,
                "horizon_seconds": float(record["future_time_seconds"][-1]),
                "canonical_camera": "T_A_from_Ct",
                "anchor": "last_observed_camera",
                "supervised_geometry": "camera_se3_and_tracked_wrist_xyz",
            },
        }
        if self.text_feature_dim > 0:
            sample["context_text_features"] = torch.zeros(
                self.text_feature_dim, dtype=torch.float32
            )
            sample["context_text_mask"] = torch.tensor(False)
        if self.visual_store is not None:
            sample["context_visual_features"] = self.visual_store.lookup(
                str(record["sample_id"])
            )
        elif self.base.decode_rgb:
            sample["context_images"] = _decode_video_frames(
                record["video_path"],
                record["history_indices"],
                image_size=self.base.image_size,
            )
        if self.future_visual_store is not None:
            sample["future_visual_latents"] = self.future_visual_store.lookup(
                str(record["sample_id"])
            )
        return sample
