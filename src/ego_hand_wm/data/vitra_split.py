"""Leakage-safe VITRA train/validation splits grouped by physical source video."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
_EPISODE_SUFFIX = re.compile(r"_ep_\d+$")
_MEMBER_PREFIXES = {
    "ego4d_cooking_and_cleaning": "Ego4D_",
    "ego4d_other": "Ego4D_",
    "egoexo4d": "EgoExo4D_",
    "epic": "epic_kitchens_",
    "ssv2": "somethingsomethingv2_",
}


def stable_rank(seed: int, namespace: str, value: str) -> bytes:
    """Return a stable pseudo-random ordering key independent of Python hash state."""

    return hashlib.sha256(f"{seed}\0{namespace}\0{value}".encode("utf-8")).digest()


def episode_member_identity(member_name: str) -> tuple[str, str]:
    """Recover ``(logical source, video_name)`` from a repacked VITRA member name."""

    parts = member_name.lstrip("./").split("/")
    if len(parts) < 2:
        raise ValueError(f"Malformed VITRA member name: {member_name!r}")
    source = parts[0]
    prefix = _MEMBER_PREFIXES.get(source)
    if prefix is None:
        raise ValueError(f"Unknown VITRA source in member: {member_name!r}")
    stem = Path(parts[-1]).stem
    episode_base = _EPISODE_SUFFIX.sub("", stem)
    if episode_base == stem or not episode_base.startswith(prefix):
        raise ValueError(f"Malformed VITRA episode filename: {member_name!r}")
    video_name = episode_base[len(prefix) :]
    if not video_name:
        raise ValueError(f"Empty VITRA video identifier: {member_name!r}")
    return source, video_name


@dataclass(frozen=True)
class VitraVideoSplit:
    """Loaded split manifest used by the streaming dataset.

    Training excludes every episode belonging to a held-out physical video. Validation uses a
    deterministic, bounded member subset of those videos so frequent evaluation stays cheap.
    """

    path: Path
    seed: int
    dataset_aliases: dict[str, str]
    validation_videos: dict[str, frozenset[str]]
    validation_members: dict[str, frozenset[str]]

    @classmethod
    def load(cls, path: str | Path) -> "VitraVideoSplit":
        resolved = Path(path)
        payload = json.loads(resolved.read_text(encoding="utf-8"))
        if int(payload.get("schema_version", -1)) != SCHEMA_VERSION:
            raise ValueError(f"Unsupported VITRA split schema in {resolved}")
        aliases = {str(key): str(value) for key, value in payload["dataset_aliases"].items()}
        videos = {
            str(source): frozenset(str(video) for video in values)
            for source, values in payload["validation_videos"].items()
        }
        members = {
            str(source): frozenset(str(member) for member in values)
            for source, values in payload["validation_members"].items()
        }
        if not videos or not members:
            raise ValueError(f"VITRA split is empty: {resolved}")
        return cls(
            path=resolved,
            seed=int(payload["seed"]),
            dataset_aliases=aliases,
            validation_videos=videos,
            validation_members=members,
        )

    def physical_source(self, logical_source: str) -> str:
        return self.dataset_aliases.get(logical_source, logical_source)

    def is_validation_video(self, logical_source: str, video_name: str) -> bool:
        return video_name in self.validation_videos.get(
            self.physical_source(logical_source), frozenset()
        )

    def includes(
        self,
        split: str,
        *,
        logical_source: str,
        video_name: str,
        member_name: str,
    ) -> bool:
        held_out = self.is_validation_video(logical_source, video_name)
        if split == "train":
            return not held_out
        if split == "validation":
            return held_out and member_name in self.validation_members.get(
                logical_source, frozenset()
            )
        raise ValueError(f"Unknown VITRA split: {split!r}")

    def episode_seed(self, member_name: str) -> int:
        return int.from_bytes(stable_rank(self.seed, "validation-window", member_name)[:8], "big")


def validate_aliases(config_aliases: dict[str, str], split: VitraVideoSplit) -> None:
    """Prevent the loader and manifest from disagreeing about physical-video identity."""

    if config_aliases != split.dataset_aliases:
        raise ValueError(
            "data.dataset_aliases must exactly match the VITRA split manifest aliases: "
            f"{config_aliases!r} != {split.dataset_aliases!r}"
        )
