"""Controlled, subject-disjoint forecasting dataset over official HOT3D-Clips."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from ego_hand_wm.data.adapters.hot3d_clips import canonicalize_hot3d_clip_window
from ego_hand_wm.data.trajectory_features import TrajectoryVisualFeatureStore


class Hot3DClipsForecastDataset(Dataset):
    """Load H6/K16 windows from public labeled Aria clip tar files.

    This is a controlled benchmark derived only from the official public
    training clips.  It is intentionally not called the official HOT3D test
    split because public test hand annotations are withheld.
    """

    provides_future_visual = False

    def __init__(self, config: dict[str, Any]) -> None:
        split = str(config.get("split", "train"))
        manifests = config.get("manifests")
        if manifests is None:
            manifest = Path(config["manifest"])
        else:
            try:
                manifest = Path(manifests[split])
            except KeyError as error:
                raise KeyError(f"No HOT3D-Clips manifest for split {split!r}") from error
        if not manifest.is_file():
            raise FileNotFoundError(manifest)
        self.records = [
            json.loads(line)
            for line in manifest.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not self.records:
            raise ValueError(f"HOT3D-Clips manifest is empty: {manifest}")
        manifest_splits = {str(record["split"]) for record in self.records}
        if manifest_splits != {split}:
            raise ValueError(
                f"HOT3D manifest split {manifest_splits} does not match requested {split!r}"
            )
        for record in self.records:
            if record.get("protocol") != "h6_k16_30hz":
                raise ValueError(f"Unsupported protocol in {record.get('sample_id')}")
        self.split = split
        self.decode_rgb = bool(config.get("decode_rgb", False))
        self.image_size = (
            int(config["image_size"]) if config.get("image_size") is not None else None
        )
        self.camera_stream = str(config.get("camera_stream", "214-1"))
        self.anchor = str(config.get("anchor", "last_observed"))
        self.wrist_target = str(config.get("wrist_target", "xyz"))
        self.rotate_clockwise = bool(config.get("rotate_clockwise", True))
        self.text_feature_dim = int(config.get("missing_text_feature_dim", 0))
        feature_root = config.get("visual_feature_root")
        self.visual_store = (
            TrajectoryVisualFeatureStore(
                feature_root,
                dataset="hot3d_clips_aria",
                split=self.split,
                sequence="history",
                output_dtype=config.get("visual_feature_dtype", "float32"),
            )
            if feature_root is not None
            else None
        )
        if self.visual_store is not None and self.decode_rgb:
            raise ValueError("Choose cached DINO.txt features or online HOT3D RGB, not both")
        future_feature_root = config.get("future_visual_feature_root")
        future_splits = {
            str(value) for value in config.get("future_visual_splits", (self.split,))
        }
        requested_split = str(config.get("split", "train"))
        attach_future = self.split in future_splits or requested_split in future_splits
        self.future_visual_store = (
            TrajectoryVisualFeatureStore(
                future_feature_root,
                dataset="hot3d_clips_aria",
                split=self.split,
                sequence="future",
                output_dtype=config.get("visual_feature_dtype", "float32"),
            )
            if future_feature_root is not None and attach_future
            else None
        )
        self.provides_context_visual = self.decode_rgb or self.visual_store is not None
        self.provides_future_visual = self.future_visual_store is not None

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        sample = canonicalize_hot3d_clip_window(
            record["tar_path"],
            record["history_indices"],
            record["future_indices"],
            record["history_time_seconds"],
            record["future_time_seconds"],
            tracked_hand=record["tracked_hand"],
            camera_stream=self.camera_stream,
            anchor=self.anchor,
            wrist_target=self.wrist_target,
            decode_rgb=self.decode_rgb,
            image_size=self.image_size,
            rotate_clockwise=self.rotate_clockwise,
        )
        sample["metadata"] = {
            **record,
            **sample["metadata"],
            "source_dataset": "hot3d_clips_aria",
            "horizon_seconds": float(record["future_time_seconds"][-1]),
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
        if self.future_visual_store is not None:
            sample["future_visual_latents"] = self.future_visual_store.lookup(
                str(record["sample_id"])
            )
        return sample
