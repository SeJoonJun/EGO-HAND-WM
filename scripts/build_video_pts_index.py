#!/usr/bin/env python3
"""Build validated per-video presentation-timestamp caches for VITRA.

The worker is intentionally source-agnostic: one Slurm array can point it at any of the
Ego4D, EgoExo4D, EPIC-KITCHENS, or SSv2 video roots.  Work is partitioned deterministically
by the sorted relative video path.  Every array task writes one atomic completion/failure
manifest, and an existing timestamp cache is reused only when both its metadata and source
fingerprint still match.

Packet PTS are read first, which only demuxes compressed packets.  If a container cannot
provide one reliable packet timestamp per frame, PyAV frame decoding is used as a correctness
fallback.  The fallback never converts frames to RGB arrays.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import av
import numpy as np
from av.error import FFmpegError


SCHEMA_VERSION = 2
SUPPORTED_SUFFIXES = frozenset({".mp4", ".webm"})
FINGERPRINT_SAMPLE_BYTES = 1 << 20


class TimestampExtractionError(RuntimeError):
    """The inexpensive timestamp path could not prove frame-level alignment."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument(
        "--glob",
        dest="patterns",
        action="append",
        default=None,
        help=(
            "Optional recursive Path.glob pattern; repeat to supply several. Files are still "
            "filtered case-insensitively to MP4 or WebM. Default: recursively inspect all files."
        ),
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def validate_shard(shard_id: int, num_shards: int) -> None:
    if num_shards <= 0:
        raise ValueError("num_shards must be positive")
    if shard_id < 0 or shard_id >= num_shards:
        raise ValueError(f"shard_id must be in [0, {num_shards}), got {shard_id}")


def discover_videos(video_root: Path, patterns: Iterable[str] | None = None) -> list[Path]:
    if not video_root.is_dir():
        raise FileNotFoundError(f"Video root does not exist: {video_root}")
    requested_patterns = list(patterns or ("**/*",))
    candidates: set[Path] = set()
    for pattern in requested_patterns:
        candidates.update(
            path
            for path in video_root.glob(pattern)
            if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
        )
    videos = sorted(candidates, key=lambda path: path.relative_to(video_root).as_posix())
    if not videos:
        joined = ", ".join(requested_patterns)
        raise FileNotFoundError(
            f"No MP4/MP4-case-insensitive or WebM videos under {video_root} matching {joined}"
        )
    return videos


def ensure_unique_video_stems(videos: Iterable[Path], video_root: Path) -> None:
    """The VITRA annotation resolver addresses timestamp arrays by ``video_name``/stem."""
    owners: dict[str, Path] = {}
    collisions: list[tuple[str, Path, Path]] = []
    for video in videos:
        previous = owners.setdefault(video.stem, video)
        if previous != video:
            collisions.append((video.stem, previous, video))
    if collisions:
        examples = "; ".join(
            f"{stem}: {first.relative_to(video_root)} vs {second.relative_to(video_root)}"
            for stem, first, second in collisions[:5]
        )
        raise ValueError(
            "Video stems are not unique, so `<video_name>.npy` would collide. " + examples
        )


def assigned_videos(videos: list[Path], shard_id: int, num_shards: int) -> list[Path]:
    validate_shard(shard_id, num_shards)
    return videos[shard_id::num_shards]


def _sampled_content_digest(path: Path, sample_bytes: int = FINGERPRINT_SAMPLE_BYTES) -> str:
    size = path.stat().st_size
    digest = hashlib.sha256()
    digest.update(str(size).encode("ascii"))
    with path.open("rb") as handle:
        digest.update(handle.read(sample_bytes))
        if size > sample_bytes:
            handle.seek(max(0, size - sample_bytes))
            digest.update(handle.read(sample_bytes))
    return digest.hexdigest()


def source_fingerprint(path: Path, video_root: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "relative_path": path.relative_to(video_root).as_posix(),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sampled_sha256": _sampled_content_digest(path),
    }


def validate_timestamps(values: np.ndarray) -> np.ndarray:
    timestamps = np.asarray(values, dtype=np.float64)
    if timestamps.ndim != 1:
        raise ValueError(f"Timestamps must be one-dimensional, got {timestamps.shape}")
    if len(timestamps) < 2:
        raise ValueError("A video timestamp array must contain at least two frames")
    if not np.isfinite(timestamps).all():
        raise ValueError("Video timestamps contain NaN or Inf")
    if np.any(np.diff(timestamps) <= 0):
        raise ValueError("Video timestamps are not strictly increasing")
    return timestamps


def _expected_frame_count(stream: Any) -> int | None:
    declared = int(stream.frames or 0)
    if declared > 0:
        return declared
    if stream.duration is None or stream.average_rate is None:
        return None
    estimate = float(stream.duration * stream.time_base * stream.average_rate)
    rounded = int(round(estimate))
    return rounded if rounded > 0 and abs(estimate - rounded) <= 0.51 else None


def _packet_pts_seconds(path: Path) -> np.ndarray:
    values: list[float] = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        expected = _expected_frame_count(stream)
        for packet in container.demux(stream):
            if packet.size <= 0:
                continue
            if packet.pts is None:
                raise TimestampExtractionError(f"Video packet lacks PTS: {path}")
            time_base = packet.time_base or stream.time_base
            if time_base is None:
                raise TimestampExtractionError(f"Video packet lacks a time base: {path}")
            values.append(float(packet.pts * time_base))
    if expected is not None and len(values) != expected:
        raise TimestampExtractionError(
            f"Packet count {len(values)} does not equal declared frame count {expected}: {path}"
        )
    try:
        return validate_timestamps(np.sort(np.asarray(values, dtype=np.float64)))
    except ValueError as error:
        raise TimestampExtractionError(str(error)) from error


def _decoded_frame_pts_seconds(path: Path) -> np.ndarray:
    values: list[float | None] = []
    average_rate = 0.0
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        average_rate = float(stream.average_rate) if stream.average_rate else 0.0
        for frame in container.decode(stream):
            if frame.pts is None:
                values.append(None)
            else:
                time_base = frame.time_base or stream.time_base
                if time_base is None:
                    raise ValueError(f"Decoded frame lacks a time base: {path}")
                values.append(float(frame.pts * time_base))
    if values and all(value is None for value in values):
        if average_rate <= 0:
            raise ValueError(f"Frames have no PTS and the stream has no average rate: {path}")
        return validate_timestamps(np.arange(len(values), dtype=np.float64) / average_rate)
    if any(value is None for value in values):
        raise ValueError(f"Only a subset of decoded frames has PTS; refusing mixed timing: {path}")
    return validate_timestamps(np.asarray(values, dtype=np.float64))


def frame_pts_seconds(path: Path) -> tuple[np.ndarray, str]:
    """Return presentation-order seconds and the extraction method used.

    Demuxed packet timestamps avoid frame/RGB reconstruction. Containers for which packet count
    or PTS integrity cannot be proven automatically fall back to decoded AVFrames. No path calls
    ``to_ndarray`` or performs a color conversion.
    """
    try:
        return _packet_pts_seconds(path), "packet_pts"
    except (TimestampExtractionError, FFmpegError, OSError):
        return _decoded_frame_pts_seconds(path), "decoded_frame_pts"


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_npy(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.save(handle, values, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def cache_paths(output_dir: Path, video: Path) -> tuple[Path, Path]:
    return output_dir / f"{video.stem}.npy", output_dir / f"{video.stem}.meta.json"


def _valid_existing_cache(
    pts_path: Path,
    metadata_path: Path,
    *,
    dataset_name: str,
    fingerprint: dict[str, Any],
) -> bool:
    if not pts_path.is_file() or not metadata_path.is_file():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("schema_version") != SCHEMA_VERSION:
            return False
        if metadata.get("dataset_name") != dataset_name:
            return False
        if metadata.get("source_fingerprint") != fingerprint:
            return False
        timestamps = validate_timestamps(np.load(pts_path, allow_pickle=False, mmap_mode="r"))
        if int(metadata.get("frame_count", -1)) != len(timestamps):
            return False
        if float(metadata.get("first_pts_seconds", float("nan"))) != float(timestamps[0]):
            return False
        if float(metadata.get("last_pts_seconds", float("nan"))) != float(timestamps[-1]):
            return False
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False
    return True


TimestampReader = Callable[[Path], tuple[np.ndarray, str] | np.ndarray]


def build_or_validate_cache(
    video: Path,
    *,
    video_root: Path,
    output_dir: Path,
    dataset_name: str,
    timestamp_reader: TimestampReader = frame_pts_seconds,
    force: bool = False,
) -> dict[str, Any]:
    pts_path, metadata_path = cache_paths(output_dir, video)
    before = source_fingerprint(video, video_root)
    if not force and _valid_existing_cache(
        pts_path, metadata_path, dataset_name=dataset_name, fingerprint=before
    ):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        return {
            "video": before["relative_path"],
            "cache": pts_path.name,
            "status": "validated_skip",
            "frame_count": int(metadata["frame_count"]),
            "method": str(metadata["timestamp_method"]),
        }

    extracted = timestamp_reader(video)
    if isinstance(extracted, tuple):
        values, method = extracted
    else:
        values, method = extracted, "injected_reader"
    timestamps = validate_timestamps(values)
    after = source_fingerprint(video, video_root)
    if after != before:
        raise RuntimeError(f"Source video changed while timestamps were being built: {video}")

    atomic_write_npy(pts_path, timestamps)
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "dataset_name": dataset_name,
        "source_fingerprint": after,
        "timestamp_method": method,
        "frame_count": int(len(timestamps)),
        "first_pts_seconds": float(timestamps[0]),
        "last_pts_seconds": float(timestamps[-1]),
        "created_at_utc": utc_now(),
    }
    atomic_write_json(metadata_path, metadata)
    return {
        "video": after["relative_path"],
        "cache": pts_path.name,
        "status": "rebuilt",
        "frame_count": int(len(timestamps)),
        "method": method,
    }


def shard_manifest_path(
    output_dir: Path, shard_id: int, num_shards: int
) -> Path:
    return output_dir / "_manifests" / f"pts-{shard_id:05d}-of-{num_shards:05d}.json"


def run_index(
    *,
    video_root: Path,
    output_root: Path,
    dataset_name: str,
    patterns: Iterable[str] | None,
    shard_id: int,
    num_shards: int,
    limit: int | None = None,
    force: bool = False,
    timestamp_reader: TimestampReader = frame_pts_seconds,
) -> dict[str, Any]:
    validate_shard(shard_id, num_shards)
    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative")
    output_dir = output_root / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = shard_manifest_path(output_dir, shard_id, num_shards)
    started = utc_now()
    records: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    discovered_count = 0
    assigned_count = 0
    try:
        videos = discover_videos(video_root, patterns)
        ensure_unique_video_stems(videos, video_root)
        if limit is not None:
            videos = videos[:limit]
        discovered_count = len(videos)
        work = assigned_videos(videos, shard_id, num_shards)
        assigned_count = len(work)
        for index, video in enumerate(work, start=1):
            try:
                record = build_or_validate_cache(
                    video,
                    video_root=video_root,
                    output_dir=output_dir,
                    dataset_name=dataset_name,
                    timestamp_reader=timestamp_reader,
                    force=force,
                )
                records.append(record)
                print(
                    f"[{index}/{len(work)} shard {shard_id}/{num_shards}] "
                    f"{record['status']} {record['video']} ({record['frame_count']} frames)",
                    flush=True,
                )
            except Exception as error:  # keep the shard running and publish all failures
                failure = {
                    "video": video.relative_to(video_root).as_posix(),
                    "error_type": type(error).__name__,
                    "message": str(error),
                }
                failures.append(failure)
                print(f"[FAIL] {json.dumps(failure, sort_keys=True)}", file=sys.stderr, flush=True)
    except Exception as error:
        failures.append(
            {
                "video": "<discovery>",
                "error_type": type(error).__name__,
                "message": str(error),
            }
        )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "dataset_name": dataset_name,
        "video_root": str(video_root),
        "patterns": list(patterns) if patterns is not None else None,
        "shard_id": shard_id,
        "num_shards": num_shards,
        "started_at_utc": started,
        "finished_at_utc": utc_now(),
        "discovered_videos": discovered_count,
        "assigned_videos": assigned_count,
        "successful_videos": len(records),
        "validated_skips": sum(record["status"] == "validated_skip" for record in records),
        "rebuilt_videos": sum(record["status"] == "rebuilt" for record in records),
        "failure_count": len(failures),
        "complete": not failures and len(records) == assigned_count,
        "records": records,
        "failures": failures,
    }
    atomic_write_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "complete": manifest["complete"],
                "successful_videos": len(records),
                "failures": len(failures),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return manifest


def main() -> None:
    args = parse_args()
    manifest = run_index(
        video_root=args.video_root,
        output_root=args.output_root,
        dataset_name=args.dataset_name,
        patterns=args.patterns,
        shard_id=args.shard_id,
        num_shards=args.num_shards,
        limit=args.limit,
        force=args.force,
    )
    if not manifest["complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
