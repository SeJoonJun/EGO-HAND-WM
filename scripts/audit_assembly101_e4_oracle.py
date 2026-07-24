#!/usr/bin/env python3
"""Audit the Assembly101-e4 oracle experiment manifest and derived caches."""

from __future__ import annotations

import argparse
import json
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ego_hand_wm.anticipation.dataset import (
    GAP_STEPS,
    HISTORY_STEPS,
    ORACLE_STEPS,
    geometry_cache_path,
    oracle_feature_cache_path,
    oracle_relative_times,
)
from ego_hand_wm.anticipation.protocol import AnticipationRecord, read_e4_anticipation_csv
from ego_hand_wm.data.adapters.assembly101 import (
    ANNOTATION_FPS,
    E4_VIDEO_STEMS,
    canonicalize_assembly101_oracle_geometry,
)
from ego_hand_wm.geometry.se3 import pose9_to_matrix


POSE_PREFIX = "assembly101_camera_and_hand_poses"
POSE_KINDS = (
    "camera_extrinsics_ego",
    "xf_transf",
    "landmarks3D",
    "hand_confidences",
    "timestamp",
)
WRIST_CONFIDENCE_THRESHOLD = 0.25


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations-root", required=True)
    parser.add_argument("--recordings-root", required=True)
    parser.add_argument("--poses-zip", required=True)
    parser.add_argument("--geometry-root", required=True)
    parser.add_argument("--feature-root", required=True)
    parser.add_argument("--output-json")
    parser.add_argument("--require-features", action="store_true")
    return parser.parse_args()


def _records_by_split(root: Path) -> dict[str, list[AnticipationRecord]]:
    return {
        split: read_e4_anticipation_csv(root / f"{split}.csv")
        for split in ("train", "validation")
    }


def _pose_recordings(archive: zipfile.ZipFile, recordings: set[str]) -> set[str]:
    names = set(archive.namelist())
    return {
        recording
        for recording in recordings
        if all(
            f"{POSE_PREFIX}/{kind}/{recording}.json" in names
            for kind in POSE_KINDS
        )
    }


def _geometry_indices(record: AnticipationRecord) -> tuple[np.ndarray, np.ndarray]:
    offsets = np.rint(oracle_relative_times() * ANNOTATION_FPS).astype(np.int64)
    indices = record.anchor_frame + offsets
    valid = np.ones((ORACLE_STEPS,), dtype=bool)
    indices[:HISTORY_STEPS] = np.maximum(indices[:HISTORY_STEPS], 0)
    execution = np.arange(HISTORY_STEPS + GAP_STEPS, ORACLE_STEPS)
    valid[execution] = indices[execution] <= record.end_frame
    indices[execution] = np.minimum(indices[execution], record.end_frame)
    return indices, valid


def _check_geometry_cache(
    path: Path,
    records: list[AnticipationRecord],
) -> dict[str, Any]:
    required = {
        "camera_world_from_camera",
        "wrist_world_from_hand",
        "landmarks_world",
        "wrist_confidence",
        "time_seconds",
        "raw_pose_frame",
    }
    with np.load(path, allow_pickle=False) as archive:
        missing = sorted(required.difference(archive.files))
        if missing:
            raise ValueError(f"{path} lacks {missing}")
        camera = np.asarray(archive["camera_world_from_camera"])
        wrist = np.asarray(archive["wrist_world_from_hand"])
        landmarks = np.asarray(archive["landmarks_world"])
        confidence = np.asarray(archive["wrist_confidence"])
        time = np.asarray(archive["time_seconds"])
        raw_frame = np.asarray(archive["raw_pose_frame"])
        frame_valid = (
            np.asarray(archive["frame_valid"], dtype=bool)
            if "frame_valid" in archive
            else np.ones((camera.shape[0],), dtype=bool)
        )
        source_frame = (
            np.asarray(archive["source_pose_frame"])
            if "source_pose_frame" in archive
            else raw_frame
        )

    steps = camera.shape[0]
    expected_shapes = {
        "camera": (steps, 4, 4),
        "wrist": (steps, 2, 4, 4),
        "landmarks": (steps, 2, 21, 3),
        "confidence": (steps, 2),
        "time": (steps,),
        "raw_frame": (steps,),
        "frame_valid": (steps,),
        "source_frame": (steps,),
    }
    values = {
        "camera": camera,
        "wrist": wrist,
        "landmarks": landmarks,
        "confidence": confidence,
        "time": time,
        "raw_frame": raw_frame,
        "frame_valid": frame_valid,
        "source_frame": source_frame,
    }
    for name, expected in expected_shapes.items():
        if values[name].shape != expected:
            raise ValueError(f"{path}: {name} is {values[name].shape}, expected {expected}")
    if not np.array_equal(raw_frame, np.arange(steps, dtype=raw_frame.dtype) * 2):
        raise ValueError(f"{path}: raw pose ids are not dense 0,2,4,...")
    if not np.array_equal(source_frame[frame_valid], raw_frame[frame_valid]):
        raise ValueError(f"{path}: valid dense frames do not map to their released ids")
    if not np.isfinite(camera).all() or not np.isfinite(wrist).all():
        raise ValueError(f"{path}: non-finite transforms")
    if not np.isfinite(confidence).all() or confidence.min() < 0 or confidence.max() > 1:
        raise ValueError(f"{path}: invalid hand confidence")
    if not np.isfinite(time).all():
        raise ValueError(f"{path}: non-finite time")
    homogeneous = np.array((0.0, 0.0, 0.0, 1.0), dtype=np.float32)
    if not np.allclose(camera[:, 3], homogeneous, atol=1e-5):
        raise ValueError(f"{path}: camera matrices are not homogeneous")
    if not np.allclose(wrist[:, :, 3], homogeneous, atol=1e-5):
        raise ValueError(f"{path}: wrist matrices are not homogeneous")

    anchor_unavailable = sum(record.anchor_frame >= steps for record in records)
    truncated_windows = sum(
        int(_geometry_indices(record)[0].max()) >= steps for record in records
    )
    anchor_wrist_valid = np.zeros((len(records), 2), dtype=bool)
    for index, record in enumerate(records):
        if record.anchor_frame < steps and frame_valid[record.anchor_frame]:
            anchor_wrist_valid[index] = (
                confidence[record.anchor_frame] >= WRIST_CONFIDENCE_THRESHOLD
            )

    # Check canonicalization at the earliest and latest target in the procedure.
    maximum_anchor_error = 0.0
    valid_geometry_steps = 0
    for record in (records[0], records[-1]):
        indices, timestamp_valid = _geometry_indices(record)
        timestamp_valid &= indices < steps
        if record.anchor_frame >= steps:
            timestamp_valid[:] = False
        indices = np.minimum(indices, steps - 1)
        timestamp_valid &= frame_valid[indices]
        canonical = canonicalize_assembly101_oracle_geometry(
            camera[indices],
            wrist[indices],
            landmarks[indices],
            confidence[indices],
            anchor_index=HISTORY_STEPS - 1,
        )
        anchor_matrix = pose9_to_matrix(canonical["camera_pose"][HISTORY_STEPS - 1])
        error = float(torch.max(torch.abs(anchor_matrix - torch.eye(4))).item())
        maximum_anchor_error = max(maximum_anchor_error, error)
        if error > 1e-4:
            raise ValueError(f"{path}: observation anchor is not identity (error={error})")
        if not torch.isfinite(canonical["wrist_pose"]).all():
            raise ValueError(f"{path}: non-finite canonical wrist pose")
        if not torch.isfinite(canonical["hand_pose"]).all():
            raise ValueError(f"{path}: non-finite canonical hand pose")
        valid_geometry_steps += int(timestamp_valid.sum())
    return {
        "frames": steps,
        "invalid_released_frames": int((~frame_valid).sum()),
        "max_anchor_identity_error": maximum_anchor_error,
        "checked_geometry_steps": valid_geometry_steps,
        "segments_with_anchor_unavailable": anchor_unavailable,
        "segments_with_truncated_window": truncated_windows,
        "segments_with_hand0_anchor": int(anchor_wrist_valid[:, 0].sum()),
        "segments_with_hand1_anchor": int(anchor_wrist_valid[:, 1].sum()),
        "segments_with_any_wrist_anchor": int(anchor_wrist_valid.any(axis=1).sum()),
        "segments_with_both_wrist_anchors": int(anchor_wrist_valid.all(axis=1).sum()),
    }


def audit(args: argparse.Namespace) -> dict[str, Any]:
    annotations_root = Path(args.annotations_root)
    recordings_root = Path(args.recordings_root)
    geometry_root = Path(args.geometry_root)
    feature_root = Path(args.feature_root)
    split_records = _records_by_split(annotations_root)
    all_records = [record for records in split_records.values() for record in records]
    recordings = {record.recording for record in all_records}
    if set(split_records["train"]) & set(split_records["validation"]):
        raise ValueError("Train and validation contain identical records")
    if {record.recording for record in split_records["train"]} & {
        record.recording for record in split_records["validation"]
    }:
        raise ValueError("Train and validation procedures overlap")

    split_report: dict[str, Any] = {}
    expected_feature_paths: set[Path] = set()
    records_by_recording: dict[str, list[AnticipationRecord]] = defaultdict(list)
    for split, records in split_records.items():
        segment_ids = [record.segment_id for record in records]
        if len(segment_ids) != len(set(segment_ids)):
            raise ValueError(f"{split} has duplicate e4 segment ids")
        if any(record.video_stem not in E4_VIDEO_STEMS for record in records):
            raise ValueError(f"{split} contains a non-e4 stream")
        if not all(0 <= int(record.verb) < 17 for record in records):
            raise ValueError(f"{split} has verb ids outside [0,16]")
        if not all(0 <= int(record.object) < 90 for record in records):
            raise ValueError(f"{split} has object ids outside [0,89]")
        if not all(0 <= int(record.action) < 1064 for record in records):
            raise ValueError(f"{split} has action ids outside [0,1063]")
        view_by_recording: dict[str, set[str]] = defaultdict(set)
        for record in records:
            view_by_recording[record.recording].add(record.video_stem)
            records_by_recording[record.recording].append(record)
            expected_feature_paths.add(
                oracle_feature_cache_path(feature_root, split, record)
            )
        if any(len(stems) != 1 for stems in view_by_recording.values()):
            raise ValueError(f"{split} maps a procedure to multiple e4 streams")
        missing_video = [
            record.video
            for record in records
            if not (
                recordings_root
                / record.recording
                / f"{record.video_stem}.mp4"
            ).is_file()
        ]
        if missing_video:
            raise FileNotFoundError(f"{split} lacks {len(missing_video)} e4 MP4s")
        split_report[split] = {
            "segments": len(records),
            "recordings": len(view_by_recording),
            "e4_serial_counts": {
                stem: sum(record.video_stem == stem for record in records)
                for stem in sorted(E4_VIDEO_STEMS)
            },
            "label_ranges": {
                "verb": [min(int(record.verb) for record in records), max(int(record.verb) for record in records)],
                "object": [min(int(record.object) for record in records), max(int(record.object) for record in records)],
                "action": [min(int(record.action) for record in records), max(int(record.action) for record in records)],
            },
        }

    with zipfile.ZipFile(args.poses_zip) as archive:
        pose_recordings = _pose_recordings(archive, recordings)
    expected_geometry_paths = {
        geometry_cache_path(geometry_root, record)
        for record in all_records
        if record.recording in pose_recordings
    }
    actual_geometry_paths = set(geometry_root.glob("*.npz"))
    if expected_geometry_paths != actual_geometry_paths:
        raise ValueError(
            "Geometry manifest mismatch: "
            f"missing={len(expected_geometry_paths - actual_geometry_paths)} "
            f"unexpected={len(actual_geometry_paths - expected_geometry_paths)}"
        )

    geometry_stats_by_recording: dict[str, dict[str, Any]] = {}
    for recording in sorted(pose_recordings):
        geometry_stats_by_recording[recording] = _check_geometry_cache(
            geometry_root / f"{recording}.npz",
            sorted(records_by_recording[recording], key=lambda record: record.anchor_frame),
        )
    geometry_stats = list(geometry_stats_by_recording.values())
    for split, records in split_records.items():
        no_pose = sum(record.recording not in pose_recordings for record in records)
        anchor_unavailable = sum(
            record.recording in pose_recordings
            and record.anchor_frame
            >= int(geometry_stats_by_recording[record.recording]["frames"])
            for record in records
        )
        truncated = sum(
            record.recording in pose_recordings
            and int(_geometry_indices(record)[0].max())
            >= int(geometry_stats_by_recording[record.recording]["frames"])
            for record in records
        )
        split_report[split]["geometry_coverage"] = {
            "segments_without_released_pose": no_pose,
            "segments_with_anchor_beyond_pose_stream": anchor_unavailable,
            "segments_with_any_window_truncation": truncated,
            "segments_with_full_geometry_window": len(records) - no_pose - truncated,
            "segments_with_hand0_anchor": sum(
                int(geometry_stats_by_recording[recording]["segments_with_hand0_anchor"])
                for recording in {record.recording for record in records}
                if recording in geometry_stats_by_recording
            ),
            "segments_with_hand1_anchor": sum(
                int(geometry_stats_by_recording[recording]["segments_with_hand1_anchor"])
                for recording in {record.recording for record in records}
                if recording in geometry_stats_by_recording
            ),
            "segments_with_any_wrist_anchor": sum(
                int(
                    geometry_stats_by_recording[recording][
                        "segments_with_any_wrist_anchor"
                    ]
                )
                for recording in {record.recording for record in records}
                if recording in geometry_stats_by_recording
            ),
            "segments_with_both_wrist_anchors": sum(
                int(
                    geometry_stats_by_recording[recording][
                        "segments_with_both_wrist_anchors"
                    ]
                )
                for recording in {record.recording for record in records}
                if recording in geometry_stats_by_recording
            ),
        }

    actual_feature_paths = set(feature_root.glob("*/*.npy"))
    unexpected_features = actual_feature_paths - expected_feature_paths
    missing_features = expected_feature_paths - actual_feature_paths
    if unexpected_features:
        raise ValueError(f"Feature cache has {len(unexpected_features)} unexpected files")
    if args.require_features and missing_features:
        raise ValueError(f"Feature cache is missing {len(missing_features)} files")
    for path in actual_feature_paths:
        value = np.load(path, mmap_mode="r", allow_pickle=False)
        if value.shape != (256, 1664) or value.dtype != np.float16:
            raise ValueError(f"{path}: expected float16 [256,1664], got {value.dtype} {value.shape}")
        if not np.isfinite(value).all():
            raise ValueError(f"{path}: non-finite V-JEPA tokens")

    return {
        "status": "complete" if not missing_features else "features_in_progress",
        "splits": split_report,
        "totals": {
            "segments": len(all_records),
            "recordings": len(recordings),
            "pose_available_recordings": len(pose_recordings),
            "pose_missing_recordings": len(recordings - pose_recordings),
        },
        "geometry": {
            "cache_files": len(actual_geometry_paths),
            "bytes": sum(path.stat().st_size for path in actual_geometry_paths),
            "invalid_released_frames": sum(
                item["invalid_released_frames"] for item in geometry_stats
            ),
            "segments_with_anchor_unavailable": sum(
                item["segments_with_anchor_unavailable"] for item in geometry_stats
            ),
            "segments_with_truncated_window": sum(
                item["segments_with_truncated_window"] for item in geometry_stats
            ),
            "segments_with_hand0_anchor": sum(
                item["segments_with_hand0_anchor"] for item in geometry_stats
            ),
            "segments_with_hand1_anchor": sum(
                item["segments_with_hand1_anchor"] for item in geometry_stats
            ),
            "segments_with_any_wrist_anchor": sum(
                item["segments_with_any_wrist_anchor"] for item in geometry_stats
            ),
            "segments_with_both_wrist_anchors": sum(
                item["segments_with_both_wrist_anchors"] for item in geometry_stats
            ),
            "maximum_anchor_identity_error": max(
                item["max_anchor_identity_error"] for item in geometry_stats
            ),
        },
        "vjepa": {
            "expected_files": len(expected_feature_paths),
            "cache_files": len(actual_feature_paths),
            "missing_files": len(missing_features),
            "bytes": sum(path.stat().st_size for path in actual_feature_paths),
        },
    }


def main() -> None:
    args = parse_args()
    report = audit(args)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
