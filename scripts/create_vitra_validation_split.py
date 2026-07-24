#!/usr/bin/env python3
"""Create a deterministic, source-stratified VITRA validation manifest.

The script reads only tar headers. It selects complete physical video/take groups for holdout,
then records a bounded episode subset for inexpensive recurring validation. All other episodes
from held-out videos remain excluded from training, preventing overlapping-frame leakage.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import tarfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from ego_hand_wm.data.vitra_split import SCHEMA_VERSION, episode_member_identity, stable_rank


DEFAULT_ALIASES = {"ego4d_other": "ego4d_cooking_and_cleaning"}
DEFAULT_MINIMUM_VIDEOS = {
    "ego4d_cooking_and_cleaning": 4,
    "ego4d_other": 4,
    "egoexo4d": 4,
    "epic": 4,
    "ssv2": 16,
}
DEFAULT_EVALUATION_EPISODES = {
    "ego4d_cooking_and_cleaning": 32,
    "ego4d_other": 32,
    "egoexo4d": 64,
    "epic": 64,
    "ssv2": 64,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-glob", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--exclusion-database", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _excluded_members(path: Path | None) -> set[str]:
    if path is None:
        return set()
    uri = f"file:{path.resolve()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        return {str(row[0]) for row in connection.execute("SELECT member FROM excluded")}


def _scan(
    shards: list[Path], excluded: set[str]
) -> tuple[dict[str, Counter[str]], dict[str, dict[str, list[str]]]]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    members: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for shard in shards:
        with tarfile.open(shard, "r:") as archive:
            for member in archive:
                name = member.name.lstrip("./")
                if not member.isfile() or not name.endswith(".npy") or name in excluded:
                    continue
                source, video = episode_member_identity(name)
                counts[source][video] += 1
                members[source][video].append(name)
    return dict(counts), {source: dict(videos) for source, videos in members.items()}


def _select_videos(
    counts: Counter[str], *, source: str, seed: int, minimum: int, episode_target: int
) -> list[str]:
    ranked = sorted(counts, key=lambda value: stable_rank(seed, source, value))
    selected: list[str] = []
    episodes = 0
    for video in ranked:
        selected.append(video)
        episodes += counts[video]
        if len(selected) >= minimum and episodes >= episode_target:
            break
    if len(selected) < minimum or episodes < episode_target:
        raise ValueError(
            f"Source {source!r} cannot satisfy {minimum} videos and {episode_target} episodes"
        )
    return selected


def _select_members(
    by_video: dict[str, list[str]], validation_videos: set[str], *, source: str, seed: int, cap: int
) -> list[str]:
    candidates = [
        member
        for video in validation_videos
        for member in by_video.get(video, ())
    ]
    return sorted(candidates, key=lambda value: stable_rank(seed, f"{source}-episode", value))[:cap]


def build_manifest(
    shards: list[Path], *, excluded: set[str], seed: int
) -> dict[str, Any]:
    counts, members = _scan(shards, excluded)
    expected = set(DEFAULT_MINIMUM_VIDEOS)
    if set(counts) != expected:
        raise ValueError(f"Unexpected logical VITRA sources: {sorted(counts)}")

    selected_by_logical: dict[str, list[str]] = {}
    for source in DEFAULT_MINIMUM_VIDEOS:
        selected_by_logical[source] = _select_videos(
            counts[source],
            source=source,
            seed=seed,
            minimum=DEFAULT_MINIMUM_VIDEOS[source],
            episode_target=DEFAULT_EVALUATION_EPISODES[source],
        )

    validation_videos: dict[str, set[str]] = defaultdict(set)
    for source, videos in selected_by_logical.items():
        physical = DEFAULT_ALIASES.get(source, source)
        validation_videos[physical].update(videos)

    validation_members: dict[str, list[str]] = {}
    logical_statistics: dict[str, dict[str, int]] = {}
    for source, source_counts in counts.items():
        physical = DEFAULT_ALIASES.get(source, source)
        held_out = validation_videos[physical]
        selected_members = _select_members(
            members[source],
            held_out,
            source=source,
            seed=seed,
            cap=DEFAULT_EVALUATION_EPISODES[source],
        )
        validation_members[source] = selected_members
        logical_statistics[source] = {
            "available_videos": len(source_counts),
            "available_episodes": sum(source_counts.values()),
            "selected_videos_before_physical_union": len(selected_by_logical[source]),
            "held_out_videos_present_in_source": sum(video in source_counts for video in held_out),
            "held_out_episodes": sum(source_counts.get(video, 0) for video in held_out),
            "evaluated_episodes": len(selected_members),
        }

    physical_statistics = {
        source: {"held_out_videos": len(videos)}
        for source, videos in sorted(validation_videos.items())
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "name": "vitra-small-validation-v1",
        "seed": seed,
        "strategy": "complete-physical-video-holdout-with-bounded-evaluation-members",
        "dataset_aliases": DEFAULT_ALIASES,
        "validation_videos": {
            source: sorted(videos) for source, videos in sorted(validation_videos.items())
        },
        "validation_members": validation_members,
        "logical_source_statistics": logical_statistics,
        "physical_source_statistics": physical_statistics,
        "excluded_members_omitted": len(excluded),
        "source_shards": {
            "glob_count": len(shards),
            "first": shards[0].name,
            "last": shards[-1].name,
        },
    }


def main() -> None:
    args = parse_args()
    import glob

    shards = [Path(path) for path in sorted(glob.glob(args.shard_glob))]
    if not shards:
        raise FileNotFoundError(f"No shards match {args.shard_glob}")
    manifest = build_manifest(
        shards,
        excluded=_excluded_members(args.exclusion_database),
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), **manifest["physical_source_statistics"]}))


if __name__ == "__main__":
    main()
