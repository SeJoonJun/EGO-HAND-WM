#!/usr/bin/env python3
"""Build leak-free H=6, K=16 window manifests for H2O and EgoPAT3D.

The script does not duplicate RGB.  Each JSONL record points to one source
video/trajectory and contains the exact 22 frame indices to decode.  Baseline
adapters must transform 3D targets into the last-observed camera frame.
"""

from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import numpy as np

from ego_hand_wm.benchmarks import FixedTrajectoryProtocol


EGOPAT3D_SPLITS: dict[str, dict[str, tuple[str, ...]]] = {
    "train": {
        "1": ("1", "2", "3", "4", "5", "6", "7"),
        "2": ("1", "2", "3", "4", "5", "6", "7"),
        "3": ("1", "2", "3", "4", "5", "6"),
        "4": ("1", "2", "3", "4", "5", "6", "7"),
        "5": ("1", "2", "3", "4", "5", "6"),
        "6": ("1", "2", "3", "4", "5", "6"),
        "7": ("1", "2", "3", "4", "5", "6", "7"),
        "9": ("1", "2", "3", "4", "5", "6", "7"),
        "10": ("1", "2", "3", "4", "5", "6", "7"),
        "11": ("1", "2", "3", "4", "5", "6", "7"),
        "12": ("1", "2", "3", "4", "5", "6", "7"),
    },
    "val": {
        "1": ("8",),
        "2": ("8",),
        "3": ("7",),
        "4": ("8",),
        "5": ("7",),
        "6": ("7",),
        "7": ("8",),
        "9": ("8",),
        "10": ("8",),
        "11": ("8",),
        "12": ("8",),
    },
    "test_seen": {
        "1": ("9", "10"),
        "2": ("9", "10"),
        "3": ("9", "10"),
        "4": ("9", "10"),
        "5": ("8", "9"),
        "6": ("8", "9"),
        "7": ("9", "10"),
        "9": ("9", "10"),
        "10": ("9", "10"),
        "11": ("9", "10"),
        "12": ("9", "10"),
    },
    "test_novel": {
        "13": ("1", "2", "3", "4", "5", "6", "7", "8", "9", "10"),
        "14": ("2", "3", "4", "5", "6", "7", "8", "9", "10"),
        "15": ("1", "2", "3", "4", "5", "6", "7", "8", "9", "10"),
    },
}


def _load_pickle(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        value = pickle.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a dictionary in {path}")
    return value


def _record(
    *,
    protocol: FixedTrajectoryProtocol,
    dataset: str,
    split: str,
    sample_id: str,
    source_group: str,
    video_path: Path,
    trajectory_path: Path,
    window_start: int,
    source_frame_offset: int = 0,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    temporal = protocol.manifest_fields(window_start)
    temporal["history_indices"] = [
        source_frame_offset + value for value in temporal["history_indices"]
    ]
    temporal["future_indices"] = [
        source_frame_offset + value for value in temporal["future_indices"]
    ]
    temporal["anchor_index"] = source_frame_offset + int(temporal["anchor_index"])
    result: dict[str, Any] = {
        "schema_version": 1,
        "protocol": "h6_k16_30hz",
        "dataset": dataset,
        "split": split,
        "sample_id": sample_id,
        "source_group": source_group,
        "video_path": str(video_path.resolve()),
        "trajectory_path": str(trajectory_path.resolve()),
        "target": "future_3d_hand_position",
        "target_coordinate_frame": "last_observed_camera",
        **temporal,
    }
    if extra:
        result.update(extra)
    return result


def iter_h2o_records(
    root: Path,
    *,
    protocol: FixedTrajectoryProtocol,
    stride: int,
    require_video: bool,
    counters: Counter[str],
) -> Iterator[dict[str, Any]]:
    base = root / "Ego3DTraj" if (root / "Ego3DTraj").is_dir() else root
    split_dir = base / "splits"
    for split in ("train", "val", "test"):
        split_file = split_dir / f"{split}.txt"
        if not split_file.is_file():
            raise FileNotFoundError(split_file)
        sample_names = [
            line.strip() for line in split_file.read_text().splitlines() if line.strip()
        ]
        for sample_name in sample_names:
            trajectory_path = base / "traj" / f"{sample_name}.pkl"
            video_path = base / "video" / f"{sample_name}.mp4"
            if not trajectory_path.is_file():
                counters[f"{split}:missing_trajectory"] += 1
                continue
            if require_video and not video_path.is_file():
                counters[f"{split}:missing_video"] += 1
                continue
            data = _load_pickle(trajectory_path)
            for hand in ("left_hand", "right_hand"):
                for segment_index, segment in enumerate(data.get(hand, ())):
                    start = int(segment["start"])
                    stated_length = int(segment["end"]) - start + 1
                    length = min(
                        stated_length,
                        len(segment["traj3d"]),
                        len(segment["cam2world"]),
                    )
                    starts = protocol.window_starts(length, stride=stride)
                    if not starts:
                        counters[f"{split}:short_segments"] += 1
                        continue
                    counters[f"{split}:eligible_segments"] += 1
                    for window_start in starts:
                        sample_id = (
                            f"h2o:{sample_name}:{hand}:{segment_index}:"
                            f"{start + window_start}"
                        )
                        yield _record(
                            protocol=protocol,
                            dataset="h2o",
                            split=split,
                            sample_id=sample_id,
                            source_group=sample_name,
                            video_path=video_path,
                            trajectory_path=trajectory_path,
                            window_start=window_start,
                            source_frame_offset=start,
                            extra={
                                "hand": hand.removesuffix("_hand"),
                                "trajectory_segment_index": segment_index,
                                "trajectory_window_start": window_start,
                                "camera_transform_key": "cam2world",
                                "trajectory_key": "traj3d",
                            },
                        )


def _record_in_split(record_name: str, record_ids: Iterable[str]) -> bool:
    return record_name.rsplit("_", 1)[-1] in record_ids


def iter_egopat3d_records(
    root: Path,
    *,
    protocol: FixedTrajectoryProtocol,
    stride: int,
    require_video: bool,
    counters: Counter[str],
) -> Iterator[dict[str, Any]]:
    base = root / "EgoPAT3D-postproc" if (root / "EgoPAT3D-postproc").is_dir() else root
    video_root = base / "video_clips_hand"
    trajectory_root = base / "trajectory_repair"
    odometry_root = base / "odometry"
    for required in (trajectory_root, odometry_root):
        if not required.is_dir():
            raise FileNotFoundError(required)

    for split, scenes in EGOPAT3D_SPLITS.items():
        for scene, record_ids in scenes.items():
            scene_trajectory = trajectory_root / scene
            if not scene_trajectory.is_dir():
                counters[f"{split}:missing_scenes"] += 1
                continue
            record_dirs = sorted(
                path
                for path in scene_trajectory.iterdir()
                if path.is_dir() and _record_in_split(path.name, record_ids)
            )
            for record_dir in record_dirs:
                for trajectory_path in sorted(record_dir.glob("*.pkl")):
                    relative = trajectory_path.relative_to(trajectory_root).with_suffix("")
                    video_path = (video_root / relative).with_suffix(".mp4")
                    odometry_path = (odometry_root / relative).with_suffix(".npy")
                    if require_video and not video_path.is_file():
                        counters[f"{split}:missing_video"] += 1
                        continue
                    if not odometry_path.is_file():
                        counters[f"{split}:missing_odometry"] += 1
                        continue
                    data = _load_pickle(trajectory_path)
                    preserve = int(data.get("num_preserve", len(data["traj3d"])))
                    odometry_length = int(np.load(odometry_path, mmap_mode="r").shape[0])
                    length = min(preserve, len(data["traj3d"]), odometry_length)
                    starts = protocol.window_starts(length, stride=stride)
                    if not starts:
                        counters[f"{split}:short_clips"] += 1
                        continue
                    counters[f"{split}:eligible_clips"] += 1
                    source_group = f"{scene}/{record_dir.name}"
                    for window_start in starts:
                        sample_id = f"egopat3d:{relative.as_posix()}:{window_start}"
                        yield _record(
                            protocol=protocol,
                            dataset="egopat3d",
                            split=split,
                            sample_id=sample_id,
                            source_group=source_group,
                            video_path=video_path,
                            trajectory_path=trajectory_path,
                            window_start=window_start,
                            extra={
                                "trajectory_window_start": window_start,
                                "odometry_path": str(odometry_path.resolve()),
                                "odometry_semantics": "camera_t_to_camera_t_minus_1",
                                "trajectory_key": "traj3d",
                            },
                        )


def _write_manifests(
    records: Iterable[dict[str, Any]],
    output_dir: Path,
    dataset: str,
) -> dict[str, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    handles: dict[str, Any] = {}
    counts: Counter[str] = Counter()
    try:
        for record in records:
            split = str(record["split"])
            if split not in handles:
                path = output_dir / f"{dataset}_{split}_h6_k16.jsonl"
                handles[split] = path.open("w", encoding="utf-8")
            handles[split].write(json.dumps(record, separators=(",", ":")) + "\n")
            counts[split] += 1
    finally:
        for handle in handles.values():
            handle.close()
    return dict(sorted(counts.items()))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("h2o", "egopat3d", "both"), default="both")
    parser.add_argument("--h2o-root", type=Path)
    parser.add_argument("--egopat3d-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--require-video", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol = FixedTrajectoryProtocol()
    counters: Counter[str] = Counter()
    summaries: dict[str, Any] = {
        "protocol": {
            "history_steps": protocol.history_steps,
            "future_steps": protocol.future_steps,
            "fps": protocol.fps,
            "history_span_seconds": protocol.history_span_seconds,
            "final_horizon_seconds": protocol.final_horizon_seconds,
            "stride": args.stride,
        }
    }

    if args.dataset in ("h2o", "both"):
        if args.h2o_root is None:
            raise ValueError("--h2o-root is required")
        records = iter_h2o_records(
            args.h2o_root,
            protocol=protocol,
            stride=args.stride,
            require_video=args.require_video,
            counters=counters,
        )
        summaries["h2o"] = _write_manifests(records, args.output_dir, "h2o")

    if args.dataset in ("egopat3d", "both"):
        if args.egopat3d_root is None:
            raise ValueError("--egopat3d-root is required")
        records = iter_egopat3d_records(
            args.egopat3d_root,
            protocol=protocol,
            stride=args.stride,
            require_video=args.require_video,
            counters=counters,
        )
        summaries["egopat3d"] = _write_manifests(records, args.output_dir, "egopat3d")

    summaries["audit"] = dict(sorted(counters.items()))
    summary_path = args.output_dir / "h6_k16_summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
