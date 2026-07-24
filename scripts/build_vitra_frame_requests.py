#!/usr/bin/env python3
"""Build a globally deduplicated, auditable VITRA frame-request index.

The CPU-only partition stage scans annotation shards into small SQLite databases.  The merge
stage deduplicates ``(dataset, video, frame_id)`` globally, applies known source exclusions,
validates every video's PTS cache, and reports the exact raw 256x256 RGB staging requirement.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sqlite3
import tarfile
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np


TRUNCATED_EGO4D_VIDEO = "c7a446bc-e16e-4e59-a027-e8449a01251a"
TRUNCATED_EGO4D_FRAMES = 15_788
MISSING_EPIC_VIDEO = "P01_19"
RGB_SIZE = 256
RGB_BYTES_PER_FRAME = RGB_SIZE * RGB_SIZE * 3


def _connect(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS frames (
            dataset TEXT NOT NULL,
            video TEXT NOT NULL,
            frame_id INTEGER NOT NULL,
            PRIMARY KEY (dataset, video, frame_id)
        ) WITHOUT ROWID;
        CREATE TABLE IF NOT EXISTS excluded (
            member TEXT PRIMARY KEY,
            dataset TEXT NOT NULL,
            video TEXT NOT NULL,
            reason TEXT NOT NULL
        ) WITHOUT ROWID;
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        ) WITHOUT ROWID;
        """
    )
    return connection


def _atomic_database_path(destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    return destination.with_name(f".{destination.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")


def _publish_database(temporary: Path, destination: Path) -> None:
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, destination)
    descriptor = os.open(destination.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _metadata(connection: sqlite3.Connection) -> dict[str, Any]:
    return {
        key: json.loads(value)
        for key, value in connection.execute("SELECT key, value FROM metadata")
    }


def _write_metadata(connection: sqlite3.Connection, values: dict[str, Any]) -> None:
    connection.executemany(
        "INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)",
        [(key, json.dumps(value, sort_keys=True)) for key, value in values.items()],
    )


def exclusion_reason(dataset: str, video: str, frame_ids: np.ndarray) -> str | None:
    if dataset == "epic" and video == MISSING_EPIC_VIDEO:
        return "missing_source_video:P01_19"
    if (
        dataset in {"ego4d_cooking_and_cleaning", "ego4d_other"}
        and video == TRUNCATED_EGO4D_VIDEO
        and int(frame_ids[-1]) >= TRUNCATED_EGO4D_FRAMES
    ):
        return f"truncated_source_frame>={TRUNCATED_EGO4D_FRAMES}"
    return None


def _load_episode(payload: bytes, member_name: str) -> tuple[str, str, np.ndarray, dict[str, Any]]:
    try:
        episode = np.load(io.BytesIO(payload), allow_pickle=True).item()
    except (OSError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid VITRA annotation: {member_name}") from error
    if not isinstance(episode, dict):
        raise ValueError(f"VITRA annotation is not a dictionary: {member_name}")
    dataset = member_name.split("/", 1)[0]
    video = str(episode.get("video_name", ""))
    frame_ids = np.asarray(episode.get("video_decode_frame"), dtype=np.int64)
    if (
        not dataset
        or not video
        or frame_ids.ndim != 1
        or len(frame_ids) == 0
        or np.any(frame_ids < 0)
        or np.any(np.diff(frame_ids) <= 0)
    ):
        raise ValueError(f"Invalid video/frame identity: {member_name}")
    return dataset, video, frame_ids, episode


def build_partition(
    annotation_shards: Iterable[str | Path],
    output_path: str | Path,
    *,
    partition_id: int,
    num_partitions: int,
) -> dict[str, Any]:
    if num_partitions <= 0 or partition_id < 0 or partition_id >= num_partitions:
        raise ValueError("partition_id must lie in [0, num_partitions)")
    shards = [Path(path) for path in sorted(annotation_shards)]
    assigned = shards[partition_id::num_partitions]
    if not assigned:
        raise ValueError(f"Frame-request partition {partition_id} has no annotation shards")
    destination = Path(output_path)
    temporary = _atomic_database_path(destination)
    stats: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    source_episodes: Counter[str] = Counter()
    try:
        connection = _connect(temporary)
        with connection:
            for shard in assigned:
                with tarfile.open(shard, "r:*") as archive:
                    for member in archive:
                        if not member.isfile() or not member.name.endswith(".npy"):
                            continue
                        extracted = archive.extractfile(member)
                        if extracted is None:
                            raise ValueError(f"Cannot read annotation member: {member.name}")
                        dataset, video, frame_ids, _ = _load_episode(
                            extracted.read(), member.name
                        )
                        stats["episodes_total"] += 1
                        source_episodes[dataset] += 1
                        reason = exclusion_reason(dataset, video, frame_ids)
                        if reason is not None:
                            connection.execute(
                                "INSERT INTO excluded(member,dataset,video,reason) VALUES (?,?,?,?)",
                                (member.name, dataset, video, reason),
                            )
                            stats["episodes_excluded"] += 1
                            reasons[reason] += 1
                            continue
                        stats["episodes_kept"] += 1
                        stats["frame_occurrences_kept"] += len(frame_ids)
                        connection.executemany(
                            "INSERT OR IGNORE INTO frames(dataset,video,frame_id) VALUES (?,?,?)",
                            ((dataset, video, int(frame_id)) for frame_id in frame_ids),
                        )
            unique_frames = int(
                connection.execute("SELECT COUNT(*) FROM frames").fetchone()[0]
            )
            values = {
                "complete": True,
                "partition_id": partition_id,
                "num_partitions": num_partitions,
                "assigned_shards": [path.name for path in assigned],
                "stats": dict(stats),
                "exclusion_reasons": dict(reasons),
                "source_episodes": dict(source_episodes),
                "unique_frames_within_partition": unique_frames,
            }
            _write_metadata(connection, values)
        connection.close()
        _publish_database(temporary, destination)
        return values
    finally:
        temporary.unlink(missing_ok=True)


def _parse_aliases(values: Iterable[str]) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"PTS alias must be DATASET=PTS_DATASET, got {value!r}")
        dataset, pts_dataset = value.split("=", 1)
        if not dataset or not pts_dataset or dataset in aliases:
            raise ValueError(f"Invalid or duplicate PTS alias: {value!r}")
        aliases[dataset] = pts_dataset
    return aliases


def merge_partitions(
    partition_paths: Iterable[str | Path],
    output_path: str | Path,
    *,
    pts_root: str | Path,
    pts_aliases: dict[str, str] | None = None,
) -> dict[str, Any]:
    parts = [Path(path) for path in sorted(partition_paths)]
    if not parts:
        raise ValueError("No request-index partitions were supplied")
    destination = Path(output_path)
    temporary = _atomic_database_path(destination)
    pts_root = Path(pts_root)
    aliases = dict(pts_aliases or {})
    aggregate: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    source_episodes: Counter[str] = Counter()
    seen_partitions: set[int] = set()
    expected_partitions: int | None = None
    try:
        connection = _connect(temporary)
        connection.executescript(
            """
            CREATE TABLE physical_frames (
                physical_dataset TEXT NOT NULL,
                video TEXT NOT NULL,
                frame_id INTEGER NOT NULL,
                PRIMARY KEY (physical_dataset, video, frame_id)
            ) WITHOUT ROWID;
            CREATE TABLE videos (
                physical_dataset TEXT NOT NULL,
                video TEXT NOT NULL,
                requested_frames INTEGER NOT NULL,
                minimum_frame INTEGER NOT NULL,
                maximum_frame INTEGER NOT NULL,
                pts_frames INTEGER NOT NULL,
                PRIMARY KEY (physical_dataset, video)
            ) WITHOUT ROWID
            """
        )
        for index, part in enumerate(parts):
            if not part.is_file():
                raise FileNotFoundError(f"Missing request-index partition: {part}")
            part_connection = sqlite3.connect(f"file:{part}?mode=ro", uri=True)
            metadata = _metadata(part_connection)
            part_connection.close()
            if not metadata.get("complete", False):
                raise ValueError(f"Incomplete request-index partition: {part}")
            partition_id = int(metadata["partition_id"])
            num_partitions = int(metadata["num_partitions"])
            if expected_partitions is None:
                expected_partitions = num_partitions
            if num_partitions != expected_partitions or partition_id in seen_partitions:
                raise ValueError("Request-index partition IDs are inconsistent or duplicated")
            seen_partitions.add(partition_id)
            aggregate.update(metadata["stats"])
            reasons.update(metadata["exclusion_reasons"])
            source_episodes.update(metadata["source_episodes"])
            schema = f"part_{index}"
            connection.execute(f"ATTACH DATABASE ? AS {schema}", (str(part),))
            with connection:
                connection.execute(
                    f"INSERT OR IGNORE INTO frames SELECT dataset,video,frame_id FROM {schema}.frames"
                )
                connection.execute(
                    f"INSERT INTO excluded SELECT member,dataset,video,reason FROM {schema}.excluded"
                )
            connection.execute(f"DETACH DATABASE {schema}")
        if expected_partitions is None or seen_partitions != set(range(expected_partitions)):
            raise ValueError(
                f"Expected partitions 0..{expected_partitions - 1 if expected_partitions else -1}, "
                f"found {sorted(seen_partitions)}"
            )

        logical_source_unique_frames = {
            dataset: int(count)
            for dataset, count in connection.execute(
                "SELECT dataset,COUNT(*) FROM frames GROUP BY dataset ORDER BY dataset"
            )
        }
        logical_datasets = sorted(logical_source_unique_frames)
        with connection:
            for dataset in logical_datasets:
                physical_dataset = aliases.get(dataset, dataset)
                connection.execute(
                    """
                    INSERT OR IGNORE INTO physical_frames(physical_dataset,video,frame_id)
                    SELECT ?,video,frame_id FROM frames WHERE dataset=?
                    """,
                    (physical_dataset, dataset),
                )

        physical_unique_frames: Counter[str] = Counter()
        physical_videos: Counter[str] = Counter()
        video_rows = connection.execute(
            """
            SELECT physical_dataset,video,COUNT(*),MIN(frame_id),MAX(frame_id)
            FROM physical_frames GROUP BY physical_dataset,video ORDER BY physical_dataset,video
            """
        ).fetchall()
        with connection:
            for physical_dataset, video, count, minimum, maximum in video_rows:
                pts_path = pts_root / physical_dataset / f"{video}.npy"
                if not pts_path.is_file():
                    raise FileNotFoundError(
                        f"Missing PTS cache for {physical_dataset}/{video}: {pts_path}"
                    )
                pts = np.load(pts_path, allow_pickle=False, mmap_mode="r")
                if pts.ndim != 1 or len(pts) == 0 or int(maximum) >= len(pts):
                    raise ValueError(
                        f"Requested frame {maximum} exceeds PTS cache {pts_path} ({len(pts)} frames)"
                    )
                connection.execute(
                    "INSERT INTO videos VALUES (?,?,?,?,?,?)",
                    (physical_dataset, video, count, minimum, maximum, len(pts)),
                )
                physical_unique_frames[physical_dataset] += int(count)
                physical_videos[physical_dataset] += 1

        logical_unique_frames = int(connection.execute("SELECT COUNT(*) FROM frames").fetchone()[0])
        unique_frames = int(
            connection.execute("SELECT COUNT(*) FROM physical_frames").fetchone()[0]
        )
        raw_bytes = unique_frames * RGB_BYTES_PER_FRAME
        summary = {
            "complete": True,
            "partitions": len(parts),
            "stats": dict(aggregate),
            "exclusion_reasons": dict(reasons),
            "source_episodes": dict(source_episodes),
            "logical_requested_frames": logical_unique_frames,
            "logical_source_unique_frames": logical_source_unique_frames,
            "unique_requested_frames": unique_frames,
            "physical_source_unique_frames": dict(physical_unique_frames),
            "physical_source_videos": dict(physical_videos),
            "staged_rgb": {
                "height": RGB_SIZE,
                "width": RGB_SIZE,
                "channels": 3,
                "dtype": "uint8",
                "bytes_per_frame": RGB_BYTES_PER_FRAME,
                "total_bytes": raw_bytes,
                "total_gib": raw_bytes / 2**30,
                "total_tib": raw_bytes / 2**40,
            },
            "pts_aliases": aliases,
        }
        with connection:
            _write_metadata(connection, {"complete": True, "summary": summary})
        connection.close()
        _publish_database(temporary, destination)
        success_path = destination.with_suffix(destination.suffix + ".SUCCESS.json")
        temporary_json = success_path.with_name(
            f".{success_path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        )
        try:
            temporary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
            os.replace(temporary_json, success_path)
        finally:
            temporary_json.unlink(missing_ok=True)
        return summary
    finally:
        temporary.unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    partition = commands.add_parser("partition")
    partition.add_argument("--annotation-shard", type=Path, action="append", required=True)
    partition.add_argument("--output", type=Path, required=True)
    partition.add_argument("--partition-id", type=int, required=True)
    partition.add_argument("--num-partitions", type=int, required=True)
    merge = commands.add_parser("merge")
    merge.add_argument("--partition", type=Path, action="append", required=True)
    merge.add_argument("--output", type=Path, required=True)
    merge.add_argument("--pts-root", type=Path, required=True)
    merge.add_argument("--pts-alias", action="append", default=[])
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "partition":
        result = build_partition(
            args.annotation_shard,
            args.output,
            partition_id=args.partition_id,
            num_partitions=args.num_partitions,
        )
    else:
        result = merge_partitions(
            args.partition,
            args.output,
            pts_root=args.pts_root,
            pts_aliases=_parse_aliases(args.pts_alias),
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
