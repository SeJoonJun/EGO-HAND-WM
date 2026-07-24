#!/usr/bin/env python3
"""Build compact 30 fps e4 camera/wrist caches directly from AssemblyPoses.zip."""

from __future__ import annotations

import argparse
import json
import os
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

from ego_hand_wm.anticipation.protocol import AnticipationRecord, read_e4_anticipation_csv
from ego_hand_wm.data.adapters.assembly101 import e4_pose_camera_key


POSE_PREFIX = "assembly101_camera_and_hand_poses"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poses-zip", required=True)
    parser.add_argument("--annotations", action="append", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--index", type=int, default=None, help="Process one zero-based recording")
    parser.add_argument("--shard-index", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--list-only", action="store_true")
    return parser.parse_args()


def recording_records(annotation_paths: list[str]) -> list[AnticipationRecord]:
    records: dict[str, AnticipationRecord] = {}
    for path in annotation_paths:
        for record in read_e4_anticipation_csv(path, require_labels=False):
            previous = records.get(record.recording)
            if previous is not None and previous.video_stem != record.video_stem:
                raise ValueError(f"Recording {record.recording} maps to multiple e4 cameras")
            records[record.recording] = record
    return [records[name] for name in sorted(records)]


def read_member(archive: zipfile.ZipFile, kind: str, recording: str) -> dict[str, Any]:
    member = f"{POSE_PREFIX}/{kind}/{recording}.json"
    try:
        return json.loads(archive.read(member))
    except KeyError as error:
        raise FileNotFoundError(f"Pose archive lacks {member}") from error


def _frame_keys(
    *streams: dict[str, Any],
) -> tuple[list[str], list[str], np.ndarray]:
    """Return dense 30 Hz ids, nearest released ids, and exact-frame validity."""

    common = set(streams[0])
    for stream in streams[1:]:
        common.intersection_update(stream)
    ordered = sorted((int(key) for key in common if int(key) % 2 == 0))
    if not ordered:
        raise ValueError("Assembly101 pose streams share no even 60 Hz frames")
    released = np.asarray(ordered, dtype=np.int64)
    dense = np.arange(0, released[-1] + 1, 2, dtype=np.int64)
    insertion = np.searchsorted(released, dense)
    left = released[np.clip(insertion - 1, 0, len(released) - 1)]
    right = released[np.clip(insertion, 0, len(released) - 1)]
    use_right = np.abs(right - dense) < np.abs(dense - left)
    nearest = np.where(use_right, right, left)
    valid = np.isin(dense, released)
    return [str(index) for index in dense], [str(index) for index in nearest], valid


def build_recording_cache(
    archive: zipfile.ZipFile, record: AnticipationRecord
) -> dict[str, np.ndarray]:
    cameras = read_member(archive, "camera_extrinsics_ego", record.recording)
    wrists = read_member(archive, "xf_transf", record.recording)
    landmarks = read_member(archive, "landmarks3D", record.recording)
    confidence = read_member(archive, "hand_confidences", record.recording)
    timestamps = read_member(archive, "timestamp", record.recording)
    frames, source_frames, frame_valid = _frame_keys(
        cameras, wrists, landmarks, confidence, timestamps
    )
    camera_key = e4_pose_camera_key(record.video_stem)
    camera = np.asarray(
        [cameras[frame][camera_key] for frame in source_frames], dtype=np.float32
    )
    wrist = np.asarray(
        [
            [wrists[frame][hand] for hand in ("0", "1")]
            for frame in source_frames
        ],
        dtype=np.float32,
    )
    conf = np.asarray(
        [
            [confidence[frame][hand] for hand in ("0", "1")]
            for frame in source_frames
        ],
        dtype=np.float32,
    )
    hand_landmarks = np.asarray(
        [
            [landmarks[frame][hand] for hand in ("0", "1")]
            for frame in source_frames
        ],
        dtype=np.float32,
    )
    released_time = np.asarray(
        [timestamps[str(frame)] for frame in sorted(map(int, set(source_frames)))],
        dtype=np.float64,
    )
    if len(released_time) > 1 and not np.all(np.diff(released_time) > 0.0):
        raise ValueError(f"Pose timestamps are not strictly increasing for {record.recording}")
    time = np.asarray([timestamps[frame] for frame in source_frames], dtype=np.float64)
    time -= time[0]
    if (
        camera.shape[1:] != (4, 4)
        or wrist.shape[1:] != (2, 4, 4)
        or hand_landmarks.shape[1:] != (2, 21, 3)
    ):
        raise ValueError(f"Unexpected transform shape for {record.recording}")
    if (
        not np.isfinite(camera).all()
        or not np.isfinite(wrist).all()
        or not np.isfinite(time).all()
    ):
        raise ValueError(f"Non-finite Assembly101 geometry in {record.recording}")
    # Released timestamps are millisecond-quantized and occasionally contain a
    # 67 ms capture-clock gap even though the pose/raw-frame ids remain contiguous.
    # Semantic annotations and MP4 access are frame-indexed, so alignment follows
    # the released even raw ids (0,2,4,...) rather than rejecting normal clock
    # jitter or accumulating it into the annotation index.
    return {
        "camera_world_from_camera": camera,
        "wrist_world_from_hand": wrist,
        # Individual landmark frames can be invalid/NaN; the adapter combines this with
        # released hand confidence to construct the hand-pose mask.
        "landmarks_world": hand_landmarks,
        "wrist_confidence": conf,
        "frame_valid": frame_valid,
        "time_seconds": time,
        "raw_pose_frame": np.asarray([int(frame) for frame in frames], dtype=np.int32),
        "source_pose_frame": np.asarray(
            [int(frame) for frame in source_frames], dtype=np.int32
        ),
    }


def atomic_save(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    records = recording_records(args.annotations)
    if args.index is not None:
        if args.shard_index is not None or args.num_shards is not None:
            raise ValueError("--index cannot be combined with sharding")
        if args.index < 0 or args.index >= len(records):
            raise IndexError(f"--index must be in [0,{len(records) - 1}]")
        records = [records[args.index]]
    elif args.shard_index is not None or args.num_shards is not None:
        if args.shard_index is None or args.num_shards is None:
            raise ValueError("Provide both --shard-index and --num-shards")
        if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
            raise ValueError("Shards require num_shards>0 and 0<=shard_index<num_shards")
        records = records[args.shard_index :: args.num_shards]
    if args.list_only:
        for record in records:
            print(f"{record.recording}\t{record.video_stem}")
        return
    output_root = Path(args.output_root)
    with zipfile.ZipFile(args.poses_zip) as archive:
        for record in records:
            output = output_root / f"{record.recording}.npz"
            if output.is_file() and not args.overwrite:
                print(f"skip\t{output}")
                continue
            try:
                arrays = build_recording_cache(archive, record)
            except FileNotFoundError as error:
                # The release has no pose streams for a small number of official
                # anticipation recordings.  Training keeps these samples and uses
                # an all-invalid geometry mask so every ablation sees one split.
                print(f"missing_pose\t{record.recording}\t{error}")
                continue
            atomic_save(output, arrays)
            print(f"wrote\t{output}\tframes={len(arrays['time_seconds'])}")


if __name__ == "__main__":
    main()
