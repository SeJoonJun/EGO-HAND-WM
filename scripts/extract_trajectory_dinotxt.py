#!/usr/bin/env python3
"""Extract VITRA-compatible DINO.txt context features for one H6/K16 manifest."""

from __future__ import annotations

import argparse
import io
import json
import os
import tarfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import av
import numpy as np
from PIL import Image

from ego_hand_wm.data.dinov3_features import (
    LocalDinoTxtVisualEncoder,
    build_dinotxt_extractor_metadata,
)
from ego_hand_wm.data.feature_shards import atomic_write_json, extractor_id
from ego_hand_wm.data.trajectory_features import TRAJECTORY_FEATURE_CONTRACT


def _records(path: Path, *, frame_field: str) -> list[dict[str, Any]]:
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        raise ValueError(f"Empty trajectory manifest: {path}")
    if any(record.get("protocol") != "h6_k16_30hz" for record in records):
        raise ValueError("Every record must use the h6_k16_30hz protocol")
    step_counts = {len(record[frame_field]) for record in records}
    if len(step_counts) != 1 or next(iter(step_counts)) <= 0:
        raise ValueError(f"Every record must have the same nonzero {frame_field} length")
    if len({record["sample_id"] for record in records}) != len(records):
        raise ValueError("Trajectory manifest contains duplicate sample IDs")
    return records


def _encode_video(
    path: str,
    placements: dict[int, list[tuple[int, int]]],
    *,
    output: np.ndarray,
    encoder: LocalDinoTxtVisualEncoder,
    batch_size: int,
) -> int:
    pending_frames: list[np.ndarray] = []
    pending_indices: list[int] = []
    requested = set(placements)
    final_index = max(requested)
    encoded = 0

    def flush() -> None:
        nonlocal encoded
        if not pending_frames:
            return
        features = encoder.encode(np.stack(pending_frames))
        for frame_index, value in zip(pending_indices, features, strict=True):
            for sample_index, history_index in placements[frame_index]:
                output[sample_index, history_index] = value.astype(np.float16)
        encoded += len(pending_frames)
        pending_frames.clear()
        pending_indices.clear()

    found: set[int] = set()
    with av.open(path) as container:
        for frame_index, frame in enumerate(container.decode(video=0)):
            if frame_index in requested:
                pending_frames.append(frame.to_ndarray(format="rgb24"))
                pending_indices.append(frame_index)
                found.add(frame_index)
                if len(pending_frames) >= batch_size:
                    flush()
            if frame_index >= final_index:
                break
    flush()
    missing = sorted(requested - found)
    if missing:
        raise IndexError(f"Video {path} is missing requested frames {missing[:10]}")
    return encoded


def _encode_hot3d_tar(
    path: str,
    placements: dict[int, list[tuple[int, int]]],
    *,
    output: np.ndarray,
    encoder: LocalDinoTxtVisualEncoder,
    batch_size: int,
    camera_stream: str,
) -> int:
    """Encode requested HOT3D Aria JPEGs with the canonical clockwise rotation."""

    pending_frames: list[np.ndarray] = []
    pending_indices: list[int] = []
    encoded = 0

    def flush() -> None:
        nonlocal encoded
        if not pending_frames:
            return
        features = encoder.encode(np.stack(pending_frames))
        for frame_index, value in zip(pending_indices, features, strict=True):
            for sample_index, temporal_index in placements[frame_index]:
                output[sample_index, temporal_index] = value.astype(np.float16)
        encoded += len(pending_frames)
        pending_frames.clear()
        pending_indices.clear()

    with tarfile.open(path, "r") as archive:
        for frame_index in sorted(placements):
            name = f"{frame_index:06d}.image_{camera_stream}.jpg"
            member = archive.extractfile(name)
            if member is None:
                raise FileNotFoundError(f"Missing {name} in {path}")
            image = Image.open(io.BytesIO(member.read())).convert("RGB")
            image = image.transpose(Image.Transpose.ROTATE_270)
            pending_frames.append(np.array(image, dtype=np.uint8, copy=True))
            pending_indices.append(frame_index)
            if len(pending_frames) >= batch_size:
                flush()
    flush()
    return encoded


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repo-path", type=Path, required=True)
    parser.add_argument("--weights-path", type=Path, required=True)
    parser.add_argument("--dinotxt-weights-path", type=Path, required=True)
    parser.add_argument("--bpe-path", type=Path, required=True)
    parser.add_argument("--model-name", default="dinov3_vitl16")
    parser.add_argument("--input-size", type=int, default=256)
    parser.add_argument("--spatial-grid-size", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--frame-field",
        choices=("history_indices", "future_indices"),
        default="history_indices",
    )
    parser.add_argument("--hot3d-camera-stream", default="214-1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    records = _records(args.manifest, frame_field=args.frame_field)
    datasets = {str(record["dataset"]) for record in records}
    splits = {str(record["split"]) for record in records}
    if len(datasets) != 1 or len(splits) != 1:
        raise ValueError("One extraction invocation must contain one dataset/split")
    dataset, split = next(iter(datasets)), next(iter(splits))
    directory = args.output_root / dataset
    directory.mkdir(parents=True, exist_ok=True)
    sequence = "history" if args.frame_field == "history_indices" else "future"
    prefix = split if sequence == "history" else f"{split}.future"
    success_path = directory / f"{prefix}.SUCCESS.json"
    features_path = directory / f"{prefix}.features.npy"
    ids_path = directory / f"{prefix}.sample_ids.npy"
    steps = len(records[0][args.frame_field])

    metadata = build_dinotxt_extractor_metadata(
        repo_path=args.repo_path,
        weights_path=args.weights_path,
        dinotxt_weights_path=args.dinotxt_weights_path,
        bpe_path=args.bpe_path,
        model_name=args.model_name,
        input_size=args.input_size,
        spatial_grid_size=args.spatial_grid_size,
    )
    digest = extractor_id(metadata)
    if success_path.is_file() and features_path.is_file() and ids_path.is_file():
        success = json.loads(success_path.read_text())
        features = np.load(features_path, allow_pickle=False, mmap_mode="r")
        ids = np.load(ids_path, allow_pickle=False, mmap_mode="r")
        expected = (
            len(records),
            steps,
            int(metadata["total_tokens"]),
            int(metadata["feature_dim"]),
        )
        if (
            success.get("complete") is True
            and success.get("contract") == TRAJECTORY_FEATURE_CONTRACT
            and success.get("extractor_id") == digest
            and features.shape == expected
            and features.dtype == np.float16
            and len(ids) == len(records)
        ):
            print(json.dumps({"status": "validated_skip", **success}, sort_keys=True))
            return

    encoder = LocalDinoTxtVisualEncoder(
        repo_path=args.repo_path,
        weights_path=args.weights_path,
        dinotxt_weights_path=args.dinotxt_weights_path,
        model_name=args.model_name,
        input_size=args.input_size,
        spatial_grid_size=args.spatial_grid_size,
        device=args.device,
    )
    token = f"{os.getpid()}"
    temporary_features = features_path.with_name(f".{features_path.name}.{token}.tmp")
    temporary_ids = ids_path.with_name(f".{ids_path.name}.{token}.tmp")
    output = np.lib.format.open_memmap(
        temporary_features,
        mode="w+",
        dtype=np.float16,
        shape=(len(records), steps, encoder.output_tokens, encoder.feature_dim),
    )
    max_id_length = max(len(str(record["sample_id"])) for record in records)
    sample_ids = np.lib.format.open_memmap(
        temporary_ids,
        mode="w+",
        dtype=f"<U{max_id_length}",
        shape=(len(records),),
    )
    sample_ids[:] = [str(record["sample_id"]) for record in records]
    sample_ids.flush()
    del sample_ids

    sources: dict[tuple[str, str], dict[int, list[tuple[int, int]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for sample_index, record in enumerate(records):
        if record["dataset"] == "hot3d_clips_aria":
            source = ("hot3d_tar", str(record["tar_path"]))
        else:
            source = ("video", str(record["video_path"]))
        for temporal_index, frame_index in enumerate(record[args.frame_field]):
            sources[source][int(frame_index)].append(
                (sample_index, temporal_index)
            )
    unique_frames = 0
    for source_index, ((source_kind, source_path), placements) in enumerate(
        sorted(sources.items()), start=1
    ):
        if source_kind == "hot3d_tar":
            unique_frames += _encode_hot3d_tar(
                source_path,
                placements,
                output=output,
                encoder=encoder,
                batch_size=args.batch_size,
                camera_stream=args.hot3d_camera_stream,
            )
        else:
            unique_frames += _encode_video(
                source_path,
                placements,
                output=output,
                encoder=encoder,
                batch_size=args.batch_size,
            )
        if source_index % 25 == 0 or source_index == len(sources):
            output.flush()
            print(
                json.dumps(
                    {
                        "dataset": dataset,
                        "split": split,
                        "sources": source_index,
                        "total_sources": len(sources),
                        "unique_frames": unique_frames,
                    }
                ),
                flush=True,
            )
    output.flush()
    del output
    temporary_features.replace(features_path)
    temporary_ids.replace(ids_path)
    success = {
        "complete": True,
        "contract": TRAJECTORY_FEATURE_CONTRACT,
        "dataset": dataset,
        "split": split,
        "sequence": sequence,
        "samples": len(records),
        "sources": len(sources),
        "unique_frames": unique_frames,
        f"{sequence}_steps": steps,
        "total_tokens": encoder.output_tokens,
        "feature_dim": encoder.feature_dim,
        "dtype": "float16",
        "extractor_id": digest,
        "extractor": metadata,
        "manifest": str(args.manifest.resolve()),
    }
    atomic_write_json(success_path, success)
    print(json.dumps({"status": "complete", **success}, sort_keys=True))


if __name__ == "__main__":
    main()
