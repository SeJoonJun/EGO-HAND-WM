#!/usr/bin/env python3
"""Decode globally deduplicated VITRA requests into atomic 256x256 uint8 RGB arrays."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import uuid
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Sequence

import numpy as np

from ego_hand_wm.data.dinov3_features import decode_video_frames
from ego_hand_wm.data.feature_shards import atomic_write_json


RGB_SIZE = 256
FrameReader = Callable[..., Iterator[tuple[int, np.ndarray]]]


def _parse_roots(values: Iterable[str]) -> dict[str, Path]:
    roots: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Video root must be PHYSICAL_DATASET=PATH, got {value!r}")
        dataset, path = value.split("=", 1)
        root = Path(path).expanduser().resolve()
        if not dataset or dataset in roots or not root.is_dir():
            raise ValueError(f"Invalid, duplicate, or missing video root: {value!r}")
        roots[dataset] = root
    return roots


def _request_fingerprint(path: Path) -> dict[str, int | str]:
    stat = path.stat()
    return {"path": str(path.resolve()), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _sha256_array(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).view(np.uint8)).hexdigest()


def _paths(output_root: Path, dataset: str, video: str) -> tuple[Path, Path, Path]:
    if not dataset or not video or "/" in dataset or "/" in video or video in {".", ".."}:
        raise ValueError(f"Unsafe staged RGB identity: {dataset!r}/{video!r}")
    root = output_root / dataset
    return root / f"{video}.frames.npy", root / f"{video}.rgb.npy", root / f"{video}.json"


def _source_video(
    *,
    pts_root: Path,
    video_roots: dict[str, Path],
    dataset: str,
    video: str,
) -> tuple[Path, Path, dict[str, Any]]:
    if dataset not in video_roots:
        raise KeyError(f"No video root configured for physical dataset {dataset!r}")
    pts_path = pts_root / dataset / f"{video}.npy"
    metadata_path = pts_root / dataset / f"{video}.meta.json"
    try:
        metadata = json.loads(metadata_path.read_text())
        fingerprint = metadata["source_fingerprint"]
        relative = Path(fingerprint["relative_path"])
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError(f"Missing or invalid PTS source metadata: {metadata_path}") from error
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe source relative path in {metadata_path}: {relative}")
    video_path = video_roots[dataset] / relative
    if not video_path.is_file() or not pts_path.is_file():
        raise FileNotFoundError(f"Missing video or PTS cache: {video_path}, {pts_path}")
    stat = video_path.stat()
    if stat.st_size != int(fingerprint["size_bytes"]) or stat.st_mtime_ns != int(
        fingerprint["mtime_ns"]
    ):
        raise ValueError(f"Source video changed after PTS indexing: {video_path}")
    return video_path, pts_path, fingerprint


def _valid_existing(
    frames_path: Path,
    rgb_path: Path,
    manifest_path: Path,
    *,
    frame_ids: np.ndarray,
    source_fingerprint: dict[str, Any],
    request_fingerprint: dict[str, Any],
) -> bool:
    if not frames_path.is_file() or not rgb_path.is_file() or not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text())
        stored_ids = np.load(frames_path, allow_pickle=False, mmap_mode="r")
        rgb = np.load(rgb_path, allow_pickle=False, mmap_mode="r")
        return bool(
            manifest.get("complete") is True
            and manifest.get("source_fingerprint") == source_fingerprint
            and manifest.get("request_index") == request_fingerprint
            and manifest.get("frame_ids_sha256") == _sha256_array(frame_ids)
            and np.array_equal(stored_ids, frame_ids)
            and rgb.shape == (len(frame_ids), RGB_SIZE, RGB_SIZE, 3)
            and rgb.dtype == np.uint8
            and int(manifest.get("rgb_size_bytes", -1)) == rgb_path.stat().st_size
        )
    except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError):
        return False


def stage_video(
    *,
    request_database: str | Path,
    output_root: str | Path,
    pts_root: str | Path,
    video_roots: dict[str, Path],
    dataset: str,
    video: str,
    frame_reader: FrameReader = decode_video_frames,
    force: bool = False,
) -> dict[str, Any]:
    request_database = Path(request_database)
    output_root = Path(output_root)
    pts_root = Path(pts_root)
    with sqlite3.connect(f"file:{request_database}?mode=ro", uri=True) as connection:
        frame_ids = np.asarray(
            [
                row[0]
                for row in connection.execute(
                    """
                    SELECT frame_id FROM physical_frames
                    WHERE physical_dataset=? AND video=? ORDER BY frame_id
                    """,
                    (dataset, video),
                )
            ],
            dtype=np.int64,
        )
    if len(frame_ids) == 0:
        raise ValueError(f"No requested frames for {dataset}/{video}")
    video_path, pts_path, source_fingerprint = _source_video(
        pts_root=pts_root,
        video_roots=video_roots,
        dataset=dataset,
        video=video,
    )
    pts = np.load(pts_path, allow_pickle=False, mmap_mode="r")
    if int(frame_ids[-1]) >= len(pts):
        raise IndexError(f"Requested frame exceeds PTS cache for {dataset}/{video}")
    frame_times = np.asarray(pts[frame_ids], dtype=np.float64)
    frames_path, rgb_path, manifest_path = _paths(output_root, dataset, video)
    request_fingerprint = _request_fingerprint(request_database)
    if not force and _valid_existing(
        frames_path,
        rgb_path,
        manifest_path,
        frame_ids=frame_ids,
        source_fingerprint=source_fingerprint,
        request_fingerprint=request_fingerprint,
    ):
        return {"status": "validated_skip", "dataset": dataset, "video": video, "frames": len(frame_ids)}

    frames_path.parent.mkdir(parents=True, exist_ok=True)
    token = f"{os.getpid()}.{uuid.uuid4().hex}"
    temporary_frames = frames_path.with_name(f".{frames_path.name}.{token}.tmp")
    temporary_rgb = rgb_path.with_name(f".{rgb_path.name}.{token}.tmp")
    try:
        with temporary_frames.open("wb") as handle:
            np.save(handle, frame_ids, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        output = np.lib.format.open_memmap(
            temporary_rgb,
            mode="w+",
            dtype=np.uint8,
            shape=(len(frame_ids), RGB_SIZE, RGB_SIZE, 3),
        )
        count = 0
        for expected, (frame_id, rgb) in zip(
            frame_ids,
            frame_reader(
                video_path,
                frame_ids.tolist(),
                frame_times.tolist(),
                output_size=(RGB_SIZE, RGB_SIZE),
            ),
            strict=True,
        ):
            if int(frame_id) != int(expected):
                raise ValueError(
                    f"Frame reader mismatch for {dataset}/{video}: {frame_id} != {expected}"
                )
            rgb = np.asarray(rgb)
            if rgb.shape != (RGB_SIZE, RGB_SIZE, 3) or rgb.dtype != np.uint8:
                raise ValueError(f"Invalid staged RGB frame: {rgb.shape} {rgb.dtype}")
            output[count] = rgb
            count += 1
        if count != len(frame_ids):
            raise ValueError(f"Decoded {count}/{len(frame_ids)} frames for {dataset}/{video}")
        output.flush()
        del output
        with temporary_rgb.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary_frames, frames_path)
        os.replace(temporary_rgb, rgb_path)
        manifest = {
            "complete": True,
            "status": "rebuilt",
            "dataset": dataset,
            "video": video,
            "frames": len(frame_ids),
            "frame_ids_sha256": _sha256_array(frame_ids),
            "shape": [len(frame_ids), RGB_SIZE, RGB_SIZE, 3],
            "dtype": "uint8",
            "resize": "ffmpeg_swscale_to_rgb24",
            "source_video": str(video_path),
            "source_fingerprint": source_fingerprint,
            "request_index": request_fingerprint,
            "rgb_size_bytes": rgb_path.stat().st_size,
        }
        atomic_write_json(manifest_path, manifest)
        return manifest
    finally:
        temporary_frames.unlink(missing_ok=True)
        temporary_rgb.unlink(missing_ok=True)


def assigned_videos(
    request_database: str | Path, *, worker_id: int, num_workers: int
) -> list[tuple[str, str, int]]:
    if num_workers <= 0 or worker_id < 0 or worker_id >= num_workers:
        raise ValueError("worker_id must lie in [0, num_workers)")
    with sqlite3.connect(f"file:{Path(request_database)}?mode=ro", uri=True) as connection:
        rows = [
            (str(dataset), str(video), int(frames))
            for dataset, video, frames in connection.execute(
                "SELECT physical_dataset,video,requested_frames FROM videos"
            )
        ]
    bins: list[list[tuple[str, str, int]]] = [[] for _ in range(num_workers)]
    loads = [0] * num_workers
    for row in sorted(rows, key=lambda item: (-item[2], item[0], item[1])):
        target = min(range(num_workers), key=lambda index: (loads[index], index))
        bins[target].append(row)
        loads[target] += row[2]
    return bins[worker_id]


def run_worker(
    *,
    request_database: str | Path,
    output_root: str | Path,
    pts_root: str | Path,
    video_roots: dict[str, Path],
    worker_id: int,
    num_workers: int,
    limit: int | None = None,
    skip_errors: bool = False,
) -> dict[str, Any]:
    assigned = assigned_videos(
        request_database, worker_id=worker_id, num_workers=num_workers
    )
    if limit is not None:
        assigned = assigned[:limit]
    completed = skipped = frames = 0
    failures: list[dict[str, str]] = []
    for dataset, video, expected_frames in assigned:
        try:
            result = stage_video(
                request_database=request_database,
                output_root=output_root,
                pts_root=pts_root,
                video_roots=video_roots,
                dataset=dataset,
                video=video,
            )
        except Exception as error:
            if not skip_errors:
                raise
            failure = {
                "dataset": dataset,
                "video": video,
                "error": f"{type(error).__name__}: {error}",
            }
            failures.append(failure)
            print(json.dumps({"status": "failed", **failure}), flush=True)
            continue
        if int(result["frames"]) != expected_frames:
            raise ValueError(f"Request count changed for {dataset}/{video}")
        frames += expected_frames
        if result["status"] == "validated_skip":
            skipped += 1
        else:
            completed += 1
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "dataset": dataset,
                    "video": video,
                    "frames": expected_frames,
                }
            ),
            flush=True,
        )
    manifest = {
        "complete": not failures,
        "worker_id": worker_id,
        "num_workers": num_workers,
        "assigned_videos": len(assigned),
        "decoded_videos": completed,
        "validated_skips": skipped,
        "frames": frames,
        "failed_videos": len(failures),
        "failures": failures,
    }
    atomic_write_json(
        Path(output_root) / "_workers" / f"worker-{worker_id:03d}.json", manifest
    )
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request-database", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--pts-root", type=Path, required=True)
    parser.add_argument("--video-root", action="append", default=[], required=True)
    parser.add_argument("--worker-id", type=int, required=True)
    parser.add_argument("--num-workers", type=int, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--skip-errors",
        action="store_true",
        help="Record a failed video and continue so one corrupt source cannot block a worker.",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = run_worker(
        request_database=args.request_database,
        output_root=args.output_root,
        pts_root=args.pts_root,
        video_roots=_parse_roots(args.video_root),
        worker_id=args.worker_id,
        num_workers=args.num_workers,
        limit=args.limit,
        skip_errors=args.skip_errors,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
