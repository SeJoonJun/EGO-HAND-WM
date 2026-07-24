#!/usr/bin/env python3
"""Encode globally deduplicated staged VITRA RGB frames with local DINOv3."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from ego_hand_wm.data.dinov3_features import (
    LocalDinoTxtVisualEncoder,
    build_dinotxt_extractor_metadata,
)
from ego_hand_wm.data.feature_shards import atomic_write_json, extractor_id
if __package__:
    from scripts.stage_vitra_rgb import assigned_videos
else:
    from stage_vitra_rgb import assigned_videos


def assigned_tail_assist_videos(
    request_database: str | Path,
    *,
    worker_id: int,
    num_workers: int,
    base_num_workers: int,
    tail_fraction: float,
) -> list[tuple[str, str, int]]:
    """Split the last fraction of existing worker bins among extra workers.

    The production workers traverse each greedily balanced bin from its largest
    videos toward its smallest.  Selecting from the reverse end lets short-lived
    sharing jobs help without interrupting those already-running workers.
    """
    if not 0.0 < tail_fraction < 1.0:
        raise ValueError("tail_fraction must lie in (0, 1)")
    if num_workers <= 0 or worker_id < 0 or worker_id >= num_workers:
        raise ValueError("worker_id must lie in [0, num_workers)")
    if base_num_workers <= 0:
        raise ValueError("base_num_workers must be positive")

    tail: list[tuple[str, str, int]] = []
    for base_worker_id in range(base_num_workers):
        base_bin = assigned_videos(
            request_database,
            worker_id=base_worker_id,
            num_workers=base_num_workers,
        )
        target_frames = sum(item[2] for item in base_bin) * tail_fraction
        selected_frames = 0
        for item in reversed(base_bin):
            tail.append(item)
            selected_frames += item[2]
            if selected_frames >= target_frames:
                break

    bins: list[list[tuple[str, str, int]]] = [[] for _ in range(num_workers)]
    loads = [0] * num_workers
    for item in sorted(tail, key=lambda row: (-row[2], row[0], row[1])):
        target = min(range(num_workers), key=lambda index: (loads[index], index))
        bins[target].append(item)
        loads[target] += item[2]
    return bins[worker_id]


class TokenEncoder(Protocol):
    spatial_grid_size: int
    output_tokens: int

    def encode(self, rgb_frames: np.ndarray) -> np.ndarray: ...


def _sha256_array(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).view(np.uint8)).hexdigest()


def _paths(root: Path, dataset: str, video: str) -> tuple[Path, Path, Path, Path]:
    if not dataset or not video or "/" in dataset or "/" in video:
        raise ValueError(f"Unsafe staged feature identity: {dataset!r}/{video!r}")
    rgb_dir = root / dataset
    return (
        rgb_dir / f"{video}.frames.npy",
        rgb_dir / f"{video}.rgb.npy",
        rgb_dir / f"{video}.json",
        rgb_dir,
    )


def _output_paths(root: Path, dataset: str, video: str) -> tuple[Path, Path]:
    directory = root / dataset
    return directory / f"{video}.features.npy", directory / f"{video}.json"


def _valid_existing(
    feature_path: Path,
    manifest_path: Path,
    *,
    frame_ids_sha256: str,
    frame_count: int,
    extractor_digest: str,
    spatial_tokens: int,
    feature_dim: int,
) -> bool:
    if not feature_path.is_file() or not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text())
        features = np.load(feature_path, allow_pickle=False, mmap_mode="r")
        return bool(
            manifest.get("complete") is True
            and manifest.get("frame_ids_sha256") == frame_ids_sha256
            and manifest.get("extractor_id") == extractor_digest
            and features.shape == (frame_count, spatial_tokens, feature_dim)
            and features.dtype == np.float16
            and int(manifest.get("feature_size_bytes", -1)) == feature_path.stat().st_size
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def encode_staged_video(
    *,
    rgb_root: str | Path,
    output_root: str | Path,
    dataset: str,
    video: str,
    encoder: TokenEncoder,
    extractor_metadata: dict[str, Any],
    batch_size: int,
    prefetch: bool = True,
) -> dict[str, Any]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    rgb_root = Path(rgb_root)
    output_root = Path(output_root)
    frames_path, rgb_path, rgb_manifest_path, _ = _paths(rgb_root, dataset, video)
    try:
        rgb_manifest = json.loads(rgb_manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Missing or invalid staged RGB manifest: {rgb_manifest_path}") from error
    if rgb_manifest.get("complete") is not True:
        raise ValueError(f"Staged RGB is incomplete: {rgb_manifest_path}")
    frame_ids = np.load(frames_path, allow_pickle=False, mmap_mode="r")
    rgb = np.load(rgb_path, allow_pickle=False, mmap_mode="r")
    if frame_ids.ndim != 1 or rgb.shape != (len(frame_ids), 256, 256, 3) or rgb.dtype != np.uint8:
        raise ValueError(f"Invalid staged RGB arrays for {dataset}/{video}")
    frame_digest = _sha256_array(frame_ids)
    if rgb_manifest.get("frame_ids_sha256") != frame_digest:
        raise ValueError(f"Staged RGB frame digest mismatch: {dataset}/{video}")

    digest = extractor_id(extractor_metadata)
    total_tokens = int(
        getattr(encoder, "output_tokens", int(encoder.spatial_grid_size) ** 2)
    )
    feature_dim = int(extractor_metadata.get("feature_dim", 1024))
    feature_path, manifest_path = _output_paths(output_root, dataset, video)
    if _valid_existing(
        feature_path,
        manifest_path,
        frame_ids_sha256=frame_digest,
        frame_count=len(frame_ids),
        extractor_digest=digest,
        spatial_tokens=total_tokens,
        feature_dim=feature_dim,
    ):
        return {"status": "validated_skip", "dataset": dataset, "video": video, "frames": len(frame_ids)}

    feature_path.parent.mkdir(parents=True, exist_ok=True)
    token = f"{os.getpid()}.{uuid.uuid4().hex}"
    temporary = feature_path.with_name(f".{feature_path.name}.{token}.tmp")
    try:
        output = np.lib.format.open_memmap(
            temporary,
            mode="w+",
            dtype=np.float16,
            shape=(len(frame_ids), total_tokens, feature_dim),
        )
        ranges = [
            (start, min(start + batch_size, len(frame_ids)))
            for start in range(0, len(frame_ids), batch_size)
        ]

        def load_rgb(bounds: tuple[int, int]) -> np.ndarray:
            start, end = bounds
            return np.array(rgb[start:end], dtype=np.uint8, copy=True, order="C")

        if prefetch and ranges:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                pending = pool.submit(load_rgb, ranges[0])
                for index, (start, end) in enumerate(ranges):
                    frames = pending.result()
                    if index + 1 < len(ranges):
                        pending = pool.submit(load_rgb, ranges[index + 1])
                    encoded = np.asarray(encoder.encode(frames))
                    if encoded.shape != (end - start, total_tokens, feature_dim):
                        raise ValueError(
                            f"Unexpected DINO shape for {dataset}/{video}: {encoded.shape}"
                        )
                    output[start:end] = encoded.astype(np.float16)
        else:
            for start, end in ranges:
                encoded = np.asarray(encoder.encode(np.asarray(rgb[start:end])))
                if encoded.shape != (end - start, total_tokens, feature_dim):
                    raise ValueError(
                        f"Unexpected DINO shape for {dataset}/{video}: {encoded.shape}"
                    )
                output[start:end] = encoded.astype(np.float16)
        output.flush()
        del output
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, feature_path)
        manifest = {
            "complete": True,
            "status": "rebuilt",
            "dataset": dataset,
            "video": video,
            "frames": len(frame_ids),
            "frame_ids_sha256": frame_digest,
            "shape": [len(frame_ids), total_tokens, feature_dim],
            "dtype": "float16",
            "extractor_id": digest,
            "extractor": extractor_metadata,
            "rgb_manifest": str(rgb_manifest_path),
            "feature_size_bytes": feature_path.stat().st_size,
        }
        atomic_write_json(manifest_path, manifest)
        return manifest
    finally:
        temporary.unlink(missing_ok=True)


def run_worker(args: argparse.Namespace) -> dict[str, Any]:
    started_at = time.monotonic()
    metadata = build_dinotxt_extractor_metadata(
        repo_path=args.repo_path,
        weights_path=args.weights_path,
        dinotxt_weights_path=args.dinotxt_weights_path,
        bpe_path=args.bpe_path,
        model_name=args.model_name,
        input_size=args.input_size,
        spatial_grid_size=args.spatial_grid_size,
    )
    if int(metadata["feature_dim"]) != args.feature_dim:
        raise ValueError(
            f"DINO.txt feature dimension is {metadata['feature_dim']}, got {args.feature_dim}"
        )
    encoder = LocalDinoTxtVisualEncoder(
        repo_path=args.repo_path,
        weights_path=args.weights_path,
        dinotxt_weights_path=args.dinotxt_weights_path,
        model_name=args.model_name,
        input_size=args.input_size,
        spatial_grid_size=args.spatial_grid_size,
        device=args.device,
    )
    if args.assist_base_workers is None:
        videos = assigned_videos(
            args.request_database,
            worker_id=args.worker_id,
            num_workers=args.num_workers,
        )
    else:
        videos = assigned_tail_assist_videos(
            args.request_database,
            worker_id=args.worker_id,
            num_workers=args.num_workers,
            base_num_workers=args.assist_base_workers,
            tail_fraction=args.assist_tail_fraction,
        )
    assist_preexisting = 0
    if args.assist_base_workers is not None:
        pending_videos = []
        for dataset, video, expected_frames in videos:
            feature_path, manifest_path = _output_paths(
                Path(args.output_root), dataset, video
            )
            if feature_path.is_file() and manifest_path.is_file():
                assist_preexisting += 1
            else:
                pending_videos.append((dataset, video, expected_frames))
        videos = pending_videos
    decoded = skipped = frames = deferred_frames = 0
    deferred: list[str] = []
    time_limited = False
    for video_index, (dataset, video, expected_frames) in enumerate(videos):
        if (
            args.max_runtime_seconds is not None
            and time.monotonic() - started_at >= args.max_runtime_seconds
        ):
            remaining = videos[video_index:]
            deferred.extend(
                f"{item_dataset}/{item_video}"
                for item_dataset, item_video, _ in remaining
            )
            deferred_frames += sum(item_frames for _, _, item_frames in remaining)
            time_limited = True
            break
        frames_path, rgb_path, rgb_manifest_path, _ = _paths(
            Path(args.rgb_root), dataset, video
        )
        if args.available_only and not (
            frames_path.is_file() and rgb_path.is_file() and rgb_manifest_path.is_file()
        ):
            deferred.append(f"{dataset}/{video}")
            deferred_frames += expected_frames
            continue
        result = encode_staged_video(
            rgb_root=args.rgb_root,
            output_root=args.output_root,
            dataset=dataset,
            video=video,
            encoder=encoder,
            extractor_metadata=metadata,
            batch_size=args.batch_size,
            prefetch=not args.no_prefetch,
        )
        if int(result["frames"]) != expected_frames:
            raise ValueError(f"Frame count changed for {dataset}/{video}")
        frames += expected_frames
        if result["status"] == "validated_skip":
            skipped += 1
        else:
            decoded += 1
    manifest = {
        "complete": True,
        "worker_id": args.worker_id,
        "num_workers": args.num_workers,
        "assigned_videos": len(videos),
        "processed_videos": decoded + skipped,
        "encoded_videos": decoded,
        "validated_skips": skipped,
        "deferred_videos": len(deferred),
        "deferred_frames": deferred_frames,
        "deferred_examples": deferred[:20],
        "frames": frames,
        "time_limited": time_limited,
        "elapsed_seconds": time.monotonic() - started_at,
        "assist_base_workers": args.assist_base_workers,
        "assist_tail_fraction": (
            args.assist_tail_fraction if args.assist_base_workers is not None else None
        ),
        "assist_preexisting_videos": assist_preexisting,
        "extractor_id": extractor_id(metadata),
        "total_tokens": encoder.output_tokens,
        "feature_dim": encoder.feature_dim,
    }
    atomic_write_json(
        Path(args.output_root)
        / "_workers"
        / f"{args.worker_manifest_prefix}-{args.worker_id:03d}.json",
        manifest,
    )
    return manifest


def finalize(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.output_root)
    manifests = []
    for worker_id in range(args.num_workers):
        path = root / "_workers" / f"worker-{worker_id:03d}.json"
        try:
            manifest = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"Missing worker manifest: {path}") from error
        if manifest.get("complete") is not True or int(manifest.get("num_workers", -1)) != args.num_workers:
            raise ValueError(f"Invalid worker manifest: {path}")
        manifests.append(manifest)
    deferred = sum(int(item.get("deferred_videos", 0)) for item in manifests)
    if deferred:
        raise ValueError(
            f"DINO extraction still has {deferred} deferred videos; rerun the array after RGB staging"
        )
    with sqlite3.connect(f"file:{Path(args.request_database)}?mode=ro", uri=True) as connection:
        expected_videos, expected_frames = connection.execute(
            "SELECT COUNT(*), COALESCE(SUM(requested_frames), 0) FROM videos"
        ).fetchone()
    videos = sum(int(item["processed_videos"]) for item in manifests)
    frames = sum(int(item["frames"]) for item in manifests)
    extractor_ids = {str(item["extractor_id"]) for item in manifests}
    token_counts = {int(item["total_tokens"]) for item in manifests}
    feature_dims = {int(item["feature_dim"]) for item in manifests}
    if (
        videos != expected_videos
        or frames != expected_frames
        or len(extractor_ids) != 1
        or len(token_counts) != 1
        or len(feature_dims) != 1
    ):
        raise ValueError(
            f"DINO worker totals mismatch: videos={videos}/{expected_videos}, "
            f"frames={frames}/{expected_frames}, extractors={extractor_ids}"
        )
    success = {
        "complete": True,
        "contract": "ego_hand_wm.vitra_unique_dinotxt_visual_features",
        "workers": args.num_workers,
        "videos": videos,
        "frames": frames,
        "extractor_id": next(iter(extractor_ids)),
        "token_layout": "post_head_cls_then_row_major_spatial",
        "class_tokens": 1,
        "spatial_tokens": next(iter(token_counts)) - 1,
        "total_tokens": next(iter(token_counts)),
        "feature_dim": next(iter(feature_dims)),
        "dtype": "float16",
    }
    atomic_write_json(root / "_SUCCESS", success)
    return success


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    worker = commands.add_parser("worker")
    worker.add_argument("--request-database", type=Path, required=True)
    worker.add_argument("--rgb-root", type=Path, required=True)
    worker.add_argument("--output-root", type=Path, required=True)
    worker.add_argument("--repo-path", type=Path, required=True)
    worker.add_argument("--weights-path", type=Path, required=True)
    worker.add_argument("--dinotxt-weights-path", type=Path, required=True)
    worker.add_argument("--bpe-path", type=Path, required=True)
    worker.add_argument("--model-name", default="dinov3_vitl16")
    worker.add_argument("--input-size", type=int, default=256)
    worker.add_argument("--spatial-grid-size", type=int, default=4)
    worker.add_argument("--feature-dim", type=int, default=1024)
    worker.add_argument("--batch-size", type=int, default=32)
    worker.add_argument("--no-prefetch", action="store_true")
    worker.add_argument("--device", default="cuda")
    worker.add_argument("--worker-id", type=int, required=True)
    worker.add_argument("--num-workers", type=int, required=True)
    worker.add_argument("--assist-base-workers", type=int)
    worker.add_argument("--assist-tail-fraction", type=float, default=0.15)
    worker.add_argument("--worker-manifest-prefix", default="worker")
    worker.add_argument(
        "--max-runtime-seconds",
        type=float,
        help="Stop cleanly between videos after this many wall-clock seconds",
    )
    worker.add_argument(
        "--available-only",
        action="store_true",
        help="Encode completed staged videos and defer missing ones for a later resumable pass.",
    )
    finish = commands.add_parser("finalize")
    finish.add_argument("--request-database", type=Path, required=True)
    finish.add_argument("--output-root", type=Path, required=True)
    finish.add_argument("--num-workers", type=int, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = run_worker(args) if args.command == "worker" else finalize(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
