#!/usr/bin/env python3
"""Audit genuine VITRA episode and future-horizon capacity from physical PTS timestamps."""

from __future__ import annotations

import argparse
import io
import json
import os
import tarfile
import uuid
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import numpy as np

if __package__:
    from scripts.build_vitra_frame_requests import _load_episode, exclusion_reason
else:
    from build_vitra_frame_requests import _load_episode, exclusion_reason


DATASETS = (
    "ego4d_cooking_and_cleaning",
    "ego4d_other",
    "egoexo4d",
    "epic",
    "ssv2",
)
DATASET_CODE = {name: index for index, name in enumerate(DATASETS)}
TARGET_COUNTS = (4, 8, 12, 16, 24, 32, 48, 60)
HORIZON_THRESHOLDS = (0.25, 0.5, 1.0, 2.0, 3.0, 5.0)
QUANTILES = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0)


def episode_capacity(
    episode: dict[str, Any], frame_times: np.ndarray, *, history_steps: int = 6
) -> tuple[int, float]:
    """Return maximum genuine future targets and time span allowed by current sampler rules."""
    if history_steps <= 0:
        raise ValueError("history_steps must be positive")
    length = len(episode["extrinsics"])
    times = np.asarray(frame_times, dtype=np.float64)
    if times.shape != (length,) or np.any(np.diff(times) <= 0):
        raise ValueError("frame_times must be strictly increasing and episode-aligned")
    primary = str(episode.get("anno_type", "right")).lower()
    if primary not in {"left", "right"}:
        raise ValueError(f"Invalid primary hand: {primary!r}")
    kept = np.asarray(episode[primary]["kept_frames"], dtype=bool)
    if kept.shape != (length,):
        raise ValueError("Primary kept_frames is not episode-aligned")

    best_targets = 0
    best_horizon = 0.0
    for _, frame_range in episode.get("text", {}).get(primary, []):
        start = max(int(frame_range[0]), 0)
        end = min(int(frame_range[1]), length)
        first_anchor = max(start, history_steps - 1)
        valid = np.flatnonzero(kept & (np.arange(length) >= first_anchor) & (np.arange(length) < end))
        if len(valid) < 2:
            continue
        anchor = int(valid[0])
        future = valid[valid > anchor]
        if len(future) == 0:
            continue
        targets = int(len(future))
        horizon = float(times[int(future[-1])] - times[anchor])
        if targets > best_targets or (targets == best_targets and horizon > best_horizon):
            best_targets = targets
            best_horizon = horizon
    return best_targets, best_horizon


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp.npz")
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def audit_partition(
    annotation_shards: Iterable[str | Path],
    output_path: str | Path,
    *,
    pts_root: str | Path,
    partition_id: int,
    num_partitions: int,
    history_steps: int = 6,
    pts_aliases: dict[str, str] | None = None,
) -> dict[str, Any]:
    if num_partitions <= 0 or partition_id < 0 or partition_id >= num_partitions:
        raise ValueError("partition_id must lie in [0, num_partitions)")
    shards = [Path(path) for path in sorted(annotation_shards)]
    assigned = shards[partition_id::num_partitions]
    aliases = dict(pts_aliases or {})
    pts_root = Path(pts_root)

    @lru_cache(maxsize=256)
    def video_pts(dataset: str, video: str) -> np.ndarray:
        pts_dataset = aliases.get(dataset, dataset)
        path = pts_root / pts_dataset / f"{video}.npy"
        if not path.is_file():
            raise FileNotFoundError(f"Missing PTS cache: {path}")
        return np.load(path, allow_pickle=False, mmap_mode="r")

    codes: list[int] = []
    episode_frames: list[int] = []
    episode_duration: list[float] = []
    max_targets: list[int] = []
    max_horizon: list[float] = []
    support_masks: list[int] = []
    totals: Counter[str] = Counter()
    exclusions: Counter[str] = Counter()
    for shard in assigned:
        with tarfile.open(shard, "r:*") as archive:
            for member in archive:
                if not member.isfile() or not member.name.endswith(".npy"):
                    continue
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise ValueError(f"Cannot read annotation member: {member.name}")
                dataset, video, frame_ids, episode = _load_episode(extracted.read(), member.name)
                totals[dataset] += 1
                reason = exclusion_reason(dataset, video, frame_ids)
                if reason is not None:
                    exclusions[f"{dataset}:{reason}"] += 1
                    continue
                pts = video_pts(dataset, video)
                if int(frame_ids[-1]) >= len(pts):
                    raise IndexError(f"Frame exceeds PTS cache: {member.name}")
                times = np.asarray(pts[frame_ids], dtype=np.float64)
                targets, horizon = episode_capacity(
                    episode, times, history_steps=history_steps
                )
                mask = sum(1 << index for index, count in enumerate(TARGET_COUNTS) if targets >= count)
                codes.append(DATASET_CODE[dataset])
                episode_frames.append(len(frame_ids))
                episode_duration.append(float(times[-1] - times[0]))
                max_targets.append(targets)
                max_horizon.append(horizon)
                support_masks.append(mask)
    metadata = {
        "complete": True,
        "partition_id": partition_id,
        "num_partitions": num_partitions,
        "history_steps": history_steps,
        "assigned_shards": [path.name for path in assigned],
        "source_episode_totals": dict(totals),
        "exclusions": dict(exclusions),
    }
    _atomic_npz(
        Path(output_path),
        {
            "dataset_code": np.asarray(codes, dtype=np.uint8),
            "episode_frames": np.asarray(episode_frames, dtype=np.int32),
            "episode_duration_seconds": np.asarray(episode_duration, dtype=np.float32),
            "max_future_targets": np.asarray(max_targets, dtype=np.int32),
            "max_future_horizon_seconds": np.asarray(max_horizon, dtype=np.float32),
            "target_support_mask": np.asarray(support_masks, dtype=np.uint16),
            "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True)),
        },
    )
    return metadata


def _quantiles(values: np.ndarray) -> dict[str, float]:
    if len(values) == 0:
        return {}
    result = np.quantile(values.astype(np.float64), QUANTILES)
    return {f"p{int(round(q * 100)):02d}": float(value) for q, value in zip(QUANTILES, result)}


def merge_audits(partition_paths: Iterable[str | Path], output_path: str | Path) -> dict[str, Any]:
    paths = [Path(path) for path in sorted(partition_paths)]
    if not paths:
        raise ValueError("No horizon-audit partitions supplied")
    arrays: dict[str, list[np.ndarray]] = {
        "dataset_code": [],
        "episode_frames": [],
        "episode_duration_seconds": [],
        "max_future_targets": [],
        "max_future_horizon_seconds": [],
        "target_support_mask": [],
    }
    totals: Counter[str] = Counter()
    exclusions: Counter[str] = Counter()
    partitions: set[int] = set()
    expected: int | None = None
    for path in paths:
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(np.asarray(archive["metadata_json"]).item()))
            if not metadata.get("complete", False):
                raise ValueError(f"Incomplete horizon audit: {path}")
            partition_id = int(metadata["partition_id"])
            count = int(metadata["num_partitions"])
            expected = count if expected is None else expected
            if count != expected or partition_id in partitions:
                raise ValueError("Inconsistent horizon-audit partitions")
            partitions.add(partition_id)
            totals.update(metadata["source_episode_totals"])
            exclusions.update(metadata["exclusions"])
            for key in arrays:
                arrays[key].append(np.asarray(archive[key]).copy())
    if expected is None or partitions != set(range(expected)):
        raise ValueError("Horizon-audit partition set is incomplete")
    merged = {key: np.concatenate(values) for key, values in arrays.items()}

    def report(mask: np.ndarray) -> dict[str, Any]:
        eligible = mask & (merged["max_future_targets"] > 0)
        kept_count = int(mask.sum())
        eligible_count = int(eligible.sum())
        support = {}
        for index, count in enumerate(TARGET_COUNTS):
            supported = int((eligible & ((merged["target_support_mask"] & (1 << index)) != 0)).sum())
            support[str(count)] = {
                "episodes": supported,
                "percent_of_eligible": 100.0 * supported / max(eligible_count, 1),
            }
        horizon_support = {}
        for threshold in HORIZON_THRESHOLDS:
            supported = int((eligible & (merged["max_future_horizon_seconds"] >= threshold)).sum())
            horizon_support[str(threshold)] = {
                "episodes": supported,
                "percent_of_eligible": 100.0 * supported / max(eligible_count, 1),
            }
        return {
            "kept_episodes": kept_count,
            "sampler_eligible_episodes": eligible_count,
            "sampler_eligible_percent": 100.0 * eligible_count / max(kept_count, 1),
            "episode_frames": _quantiles(merged["episode_frames"][mask]),
            "episode_duration_seconds": _quantiles(merged["episode_duration_seconds"][mask]),
            "max_future_targets": _quantiles(merged["max_future_targets"][eligible]),
            "max_future_horizon_seconds": _quantiles(
                merged["max_future_horizon_seconds"][eligible]
            ),
            "target_count_support": support,
            "horizon_support": horizon_support,
        }

    summary: dict[str, Any] = {
        "complete": True,
        "history_steps": 6,
        "annotation_episode_totals": dict(totals),
        "exclusions": dict(exclusions),
        "overall": report(np.ones(len(merged["dataset_code"]), dtype=bool)),
        "sources": {},
    }
    for code, dataset in enumerate(DATASETS):
        summary["sources"][dataset] = report(merged["dataset_code"] == code)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return summary


def _aliases(values: Iterable[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        dataset, pts_dataset = value.split("=", 1)
        result[dataset] = pts_dataset
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    partition = commands.add_parser("partition")
    partition.add_argument("--annotation-shard", type=Path, action="append", required=True)
    partition.add_argument("--output", type=Path, required=True)
    partition.add_argument("--pts-root", type=Path, required=True)
    partition.add_argument("--pts-alias", action="append", default=[])
    partition.add_argument("--partition-id", type=int, required=True)
    partition.add_argument("--num-partitions", type=int, required=True)
    partition.add_argument("--history-steps", type=int, default=6)
    merge = commands.add_parser("merge")
    merge.add_argument("--partition", type=Path, action="append", required=True)
    merge.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "partition":
        result = audit_partition(
            args.annotation_shard,
            args.output,
            pts_root=args.pts_root,
            partition_id=args.partition_id,
            num_partitions=args.num_partitions,
            history_steps=args.history_steps,
            pts_aliases=_aliases(args.pts_alias),
        )
    else:
        result = merge_audits(args.partition, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
