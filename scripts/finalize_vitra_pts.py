#!/usr/bin/env python3
"""Validate every PTS-array manifest and publish an optional dataset alias."""

from __future__ import annotations

import argparse
import json
import os
import uuid
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--num-shards", type=int, default=32)
    parser.add_argument("--expected-videos", type=int, required=True)
    parser.add_argument("--alias")
    return parser.parse_args()


def atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    args = parse_args()
    dataset_dir = args.output_root / args.dataset_name
    manifest_dir = dataset_dir / "_manifests"
    expected_paths = [
        manifest_dir / f"pts-{index:05d}-of-{args.num_shards:05d}.json"
        for index in range(args.num_shards)
    ]
    actual_paths = set(manifest_dir.glob("pts-*.json"))
    missing = [path.name for path in expected_paths if path not in actual_paths]
    unexpected = sorted(path.name for path in actual_paths.difference(expected_paths))
    if missing or unexpected:
        raise RuntimeError(f"Manifest set mismatch: missing={missing}, unexpected={unexpected}")

    documents = [json.loads(path.read_text()) for path in expected_paths]
    problems: list[str] = []
    cache_names: list[str] = []
    for index, (path, document) in enumerate(zip(expected_paths, documents, strict=True)):
        if document.get("dataset_name") != args.dataset_name:
            problems.append(f"{path.name}: wrong dataset_name")
        if document.get("shard_id") != index or document.get("num_shards") != args.num_shards:
            problems.append(f"{path.name}: wrong shard identity")
        if document.get("discovered_videos") != args.expected_videos:
            problems.append(f"{path.name}: wrong discovered_videos")
        if not document.get("complete") or document.get("failure_count") != 0:
            problems.append(f"{path.name}: incomplete or contains failures")
        if document.get("assigned_videos") != document.get("successful_videos"):
            problems.append(f"{path.name}: assigned/successful count mismatch")
        records = document.get("records", [])
        if len(records) != document.get("successful_videos"):
            problems.append(f"{path.name}: record count mismatch")
        cache_names.extend(str(record["cache"]) for record in records)

    assigned = sum(int(document["assigned_videos"]) for document in documents)
    successful = sum(int(document["successful_videos"]) for document in documents)
    failures = sum(int(document["failure_count"]) for document in documents)
    if assigned != args.expected_videos or successful != args.expected_videos or failures:
        problems.append(
            f"aggregate counts: assigned={assigned}, successful={successful}, failures={failures}"
        )
    if len(cache_names) != args.expected_videos or len(set(cache_names)) != args.expected_videos:
        problems.append("cache names are missing or duplicated")

    missing_caches: list[str] = []
    for cache_name in cache_names:
        cache_path = dataset_dir / cache_name
        metadata_path = cache_path.with_suffix(".meta.json")
        if not cache_path.is_file() or not metadata_path.is_file():
            missing_caches.append(cache_name)
            if len(missing_caches) == 20:
                break
    if missing_caches:
        problems.append(f"missing cache or metadata files (first 20): {missing_caches}")
    if problems:
        raise RuntimeError("PTS finalization failed:\n- " + "\n- ".join(problems))

    summary: dict[str, object] = {
        "complete": True,
        "dataset_name": args.dataset_name,
        "num_shards": args.num_shards,
        "videos": successful,
        "failures": failures,
    }
    atomic_write_json(dataset_dir / "_SUCCESS.json", summary)

    if args.alias:
        alias_path = args.output_root / args.alias
        if os.path.lexists(alias_path):
            if not alias_path.is_symlink() or os.readlink(alias_path) != args.dataset_name:
                raise RuntimeError(f"Refusing to replace existing alias path: {alias_path}")
        else:
            alias_path.symlink_to(args.dataset_name, target_is_directory=True)
        summary["alias"] = args.alias
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
