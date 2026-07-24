#!/usr/bin/env python3
"""Extract frozen local DINOv3 features for every other raw e4 frame (logical 30 fps)."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from ego_hand_wm.anticipation.dataset import feature_cache_path
from ego_hand_wm.anticipation.protocol import AnticipationRecord, read_e4_anticipation_csv
from ego_hand_wm.data.dinov3_features import LocalDinoV3SpatialEncoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recordings-root", required=True)
    parser.add_argument("--annotations", action="append", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--repo-path", required=True)
    parser.add_argument("--weights-path", required=True)
    parser.add_argument("--model-name", default="dinov3_vitl16")
    parser.add_argument("--input-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--index", type=int, default=None, help="Process one zero-based recording")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--list-only", action="store_true")
    return parser.parse_args()


def unique_records(annotation_paths: list[str]) -> list[AnticipationRecord]:
    records: dict[str, AnticipationRecord] = {}
    for path in annotation_paths:
        for record in read_e4_anticipation_csv(path, require_labels=False):
            records[record.recording] = record
    return [records[name] for name in sorted(records)]


def extract_video(
    video_path: Path,
    encoder: LocalDinoV3SpatialEncoder,
    *,
    batch_size: int,
) -> tuple[np.ndarray, dict[str, object]]:
    try:
        import av
    except ImportError as error:
        raise RuntimeError("PyAV is required for Assembly101 DINOv3 extraction") from error
    batches: list[np.ndarray] = []
    frames: list[np.ndarray] = []
    raw_count = 0

    def flush() -> None:
        if not frames:
            return
        encoded = encoder.encode(np.stack(frames, axis=0)).mean(axis=1)
        batches.append(np.asarray(encoded, dtype=np.float16))
        frames.clear()

    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        rate = float(stream.average_rate or 0.0)
        if abs(rate - 60.0) > 0.1:
            raise ValueError(f"Expected raw 60 fps Assembly101 e4 video, got {rate}: {video_path}")
        for raw_index, frame in enumerate(container.decode(stream)):
            raw_count = raw_index + 1
            if raw_index % 2:
                continue
            frames.append(frame.to_ndarray(format="rgb24"))
            if len(frames) == batch_size:
                flush()
    flush()
    if not batches:
        raise ValueError(f"No frames decoded from {video_path}")
    features = np.concatenate(batches, axis=0)
    return features, {
        "source_video": str(video_path),
        "source_raw_fps": 60,
        "logical_fps": 30,
        "raw_frames_decoded": raw_count,
        "logical_frames_encoded": int(features.shape[0]),
        "raw_frame_mapping": "logical_frame * 2",
        "feature_shape": list(features.shape),
        "feature_dtype": str(features.dtype),
    }


def atomic_save(path: Path, features: np.ndarray, metadata: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.save(handle, features, allow_pickle=False)
    os.replace(temporary, path)
    metadata_path = path.with_suffix(".json")
    metadata_temp = metadata_path.with_name(f".{metadata_path.name}.{os.getpid()}.tmp")
    metadata_temp.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    os.replace(metadata_temp, metadata_path)


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    records = unique_records(args.annotations)
    if args.index is not None:
        if args.index < 0 or args.index >= len(records):
            raise IndexError(f"--index must be in [0,{len(records) - 1}]")
        records = [records[args.index]]
    if args.list_only:
        for record in records:
            print(f"{record.recording}\t{record.video_stem}")
        return

    encoder = LocalDinoV3SpatialEncoder(
        repo_path=args.repo_path,
        weights_path=args.weights_path,
        model_name=args.model_name,
        input_size=args.input_size,
        spatial_grid_size=1,
        device=args.device,
    )
    recordings_root = Path(args.recordings_root)
    output_root = Path(args.output_root)
    for record in records:
        output = feature_cache_path(output_root, record)
        if output.is_file() and not args.overwrite:
            print(f"skip\t{output}")
            continue
        video = recordings_root / record.video
        if not video.is_file():
            raise FileNotFoundError(f"Missing e4 video: {video}")
        features, metadata = extract_video(video, encoder, batch_size=args.batch_size)
        metadata.update(
            {
                "model_name": args.model_name,
                "repo_path": str(Path(args.repo_path).resolve()),
                "weights_path": str(Path(args.weights_path).resolve()),
                "input_size": args.input_size,
                "pooling": "mean of DINOv3 patch tokens (spatial_grid_size=1)",
            }
        )
        atomic_save(output, features, metadata)
        print(f"wrote\t{output}\tframes={len(features)}")


if __name__ == "__main__":
    main()

