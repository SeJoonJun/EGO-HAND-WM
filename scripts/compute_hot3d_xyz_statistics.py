#!/usr/bin/env python3
"""Compute leakage-free HOT3D H6/K16 wrist-XYZ normalization statistics."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ego_hand_wm.data.adapters.hot3d_clips import _hand_world_transform, _read_json, se3_from_hot3d


@dataclass
class RunningMoments:
    """Numerically stable population moments for batches of XYZ points."""

    count: int = 0
    mean: np.ndarray | None = None
    m2: np.ndarray | None = None
    minimum: np.ndarray | None = None
    maximum: np.ndarray | None = None

    def update(self, points: np.ndarray) -> None:
        values = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        if not len(values) or not np.isfinite(values).all():
            raise ValueError("HOT3D normalization points must be nonempty and finite")
        batch_count = len(values)
        batch_mean = values.mean(axis=0)
        centered = values - batch_mean
        batch_m2 = np.square(centered).sum(axis=0)
        batch_minimum = values.min(axis=0)
        batch_maximum = values.max(axis=0)
        if self.count == 0:
            self.count = batch_count
            self.mean = batch_mean
            self.m2 = batch_m2
            self.minimum = batch_minimum
            self.maximum = batch_maximum
            return
        assert self.mean is not None and self.m2 is not None
        assert self.minimum is not None and self.maximum is not None
        total = self.count + batch_count
        delta = batch_mean - self.mean
        self.mean = self.mean + delta * (batch_count / total)
        self.m2 = self.m2 + batch_m2 + np.square(delta) * self.count * batch_count / total
        self.minimum = np.minimum(self.minimum, batch_minimum)
        self.maximum = np.maximum(self.maximum, batch_maximum)
        self.count = total

    def result(self) -> dict[str, Any]:
        if self.count == 0 or self.mean is None or self.m2 is None:
            raise ValueError("Cannot finalize empty moments")
        assert self.minimum is not None and self.maximum is not None
        return {
            "count": self.count,
            "mean_m": self.mean.tolist(),
            "std_m": np.sqrt(self.m2 / self.count).tolist(),
            "min_m": self.minimum.tolist(),
            "max_m": self.maximum.tolist(),
        }


def _load_records(manifest: Path) -> list[dict[str, Any]]:
    records = [
        json.loads(line)
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        raise ValueError(f"Empty HOT3D manifest: {manifest}")
    for record in records:
        if record.get("dataset") != "hot3d_clips_aria":
            raise ValueError(f"Unexpected dataset in {record.get('sample_id')}")
        if record.get("split") != "train" or record.get("official_source_split") != "train":
            raise ValueError("Normalization must use the controlled training split only")
        if record.get("protocol") != "h6_k16_30hz":
            raise ValueError(f"Unexpected protocol in {record.get('sample_id')}")
        if len(record["history_indices"]) != 6 or len(record["future_indices"]) != 16:
            raise ValueError(f"Unexpected H/K in {record.get('sample_id')}")
    return records


def _clip_world_geometry(
    tar_path: str, frame_indices: set[int], *, camera_stream: str
) -> tuple[dict[int, np.ndarray], dict[str, dict[int, np.ndarray]]]:
    camera_world: dict[int, np.ndarray] = {}
    wrist_world: dict[str, dict[int, np.ndarray]] = {"left": {}, "right": {}}
    with tarfile.open(tar_path, "r") as archive:
        for frame_index in sorted(frame_indices):
            key = f"{frame_index:06d}"
            cameras = _read_json(archive, f"{key}.cameras.json")
            camera_world[frame_index] = (
                se3_from_hot3d(cameras[camera_stream]["T_world_from_camera"])
                .numpy()
                .astype(np.float64)
            )
            hands = _read_json(archive, f"{key}.hands.json")
            for side in ("left", "right"):
                transform = _hand_world_transform(hands, side)
                if transform is not None:
                    wrist_world[side][frame_index] = transform.numpy().astype(np.float64)
    return camera_world, wrist_world


def compute_statistics(
    records: list[dict[str, Any]], *, camera_stream: str
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["tar_path"])].append(record)
    global_moments = RunningMoments()
    hand_moments = {"left": RunningMoments(), "right": RunningMoments()}
    completed = 0
    for tar_path, clip_records in sorted(grouped.items()):
        requested = {
            int(index)
            for record in clip_records
            for field in ("history_indices", "future_indices")
            for index in record[field]
        }
        camera_world, wrist_world = _clip_world_geometry(
            tar_path, requested, camera_stream=camera_stream
        )
        for record in clip_records:
            side = str(record["tracked_hand"])
            indices = [
                *map(int, record["history_indices"]),
                *map(int, record["future_indices"]),
            ]
            anchor_index = int(record["history_indices"][-1])
            world_to_anchor = np.linalg.inv(camera_world[anchor_index])
            try:
                poses = [wrist_world[side][index] for index in indices]
            except KeyError as error:
                raise ValueError(
                    f"Incomplete {side} wrist in {record['sample_id']}"
                ) from error
            points = np.stack(
                [(world_to_anchor @ pose)[:3, 3] for pose in poses], axis=0
            )
            global_moments.update(points)
            hand_moments[side].update(points)
        completed += 1
        if completed % 100 == 0 or completed == len(grouped):
            print(
                json.dumps(
                    {
                        "phase": "statistics",
                        "clips": completed,
                        "total_clips": len(grouped),
                        "points": global_moments.count,
                    }
                ),
                flush=True,
            )
    return {
        "normalization": global_moments.result(),
        "per_hand_audit": {
            side: moments.result() for side, moments in hand_moments.items()
        },
        "clips": len(grouped),
        "windows": len(records),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--camera-stream", default="214-1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = _load_records(args.manifest)
    report = compute_statistics(records, camera_stream=args.camera_stream)
    report.update(
        {
            "complete": True,
            "contract": "hot3d_clips_aria_h6_k16_train_xyz_population_stats_v1",
            "manifest": str(args.manifest.resolve()),
            "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
            "coordinate_frame": "last_observed_camera",
            "unit": "metre",
            "timesteps_per_window": 22,
            "estimator": "population_mean_std_ddof_0",
            "leakage_policy": "controlled_train_participants_only",
        }
    )
    expected = len(records) * 22
    if report["normalization"]["count"] != expected:
        raise AssertionError(
            f"Expected {expected} points, got {report['normalization']['count']}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "complete", **report}, sort_keys=True))


if __name__ == "__main__":
    main()
