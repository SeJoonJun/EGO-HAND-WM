#!/usr/bin/env python3
"""Build subject-disjoint H6/K16 manifests from labeled official Aria clips."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from ego_hand_wm.benchmarks import FixedTrajectoryProtocol
from ego_hand_wm.data.adapters.hot3d_clips import hot3d_clip_hand_validity


# The official labeled Aria train package contains these nine participants.
# This fixed 7/1/1 division is approximately 70/15/15 by clip count and keeps
# every recording and clip from one participant in exactly one split.
PARTICIPANT_SPLITS: dict[str, tuple[str, ...]] = {
    "train": ("P0001", "P0002", "P0003", "P0009", "P0010", "P0014", "P0015"),
    "validation": ("P0011",),
    "test": ("P0012",),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Official download root containing train_aria/ and manifest JSON files.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument(
        "--scan-workers",
        type=int,
        default=16,
        help="Parallel tar readers used for the strict hand-validity audit.",
    )
    parser.add_argument(
        "--clip-ids",
        type=int,
        nargs="*",
        help="Optional subset used for smoke tests; normal benchmark builds omit this.",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Skip absent clip tars. Never use this for a reported benchmark.",
    )
    parser.add_argument(
        "--skip-hand-validity-scan",
        action="store_true",
        help="Emit both hand records without verifying all 22 labels (debug only).",
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object in {path}")
    return value


def _participant_split(participant: str) -> str:
    matches = [split for split, values in PARTICIPANT_SPLITS.items() if participant in values]
    if len(matches) != 1:
        raise ValueError(f"Participant {participant!r} maps to {matches}, expected one split")
    return matches[0]


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    temporary.replace(path)


def _scan_one_clip(task: tuple[int, Path]) -> tuple[int, dict[str, tuple[bool, ...]]]:
    clip_id, tar_path = task
    return clip_id, hot3d_clip_hand_validity(tar_path)


def main() -> None:
    args = parse_args()
    if args.stride < 1:
        raise ValueError("--stride must be positive")
    if args.scan_workers < 1:
        raise ValueError("--scan-workers must be positive")
    split_path = args.root / "clip_splits.json"
    definition_path = args.root / "clip_definitions.json"
    official_splits = _load_json(split_path)
    definitions = _load_json(definition_path)
    official_train = {int(value) for value in official_splits["train"]["Aria"]}
    if args.clip_ids:
        requested = set(args.clip_ids)
        invalid = requested - official_train
        if invalid:
            raise ValueError(
                f"Requested clips are not official labeled Aria train clips: {invalid}"
            )
        official_train &= requested

    protocol = FixedTrajectoryProtocol()
    clip_items: list[tuple[int, dict[str, Any], str, str, Path]] = []
    missing: list[Path] = []
    for clip_id in sorted(official_train):
        definition = definitions[str(clip_id)]
        if definition["device"] != "Aria":
            raise ValueError(f"Clip {clip_id} is not Aria")
        sequence_id = str(definition["sequence_id"])
        participant = sequence_id.split("_", 1)[0]
        split = _participant_split(participant)
        tar_path = args.root / "train_aria" / f"clip-{clip_id:06d}.tar"
        if not tar_path.is_file():
            missing.append(tar_path)
            continue
        clip_items.append((clip_id, definition, participant, split, tar_path))

    if missing and not args.allow_missing:
        examples = ", ".join(str(path) for path in missing[:3])
        raise FileNotFoundError(
            f"Missing {len(missing)} official Aria tar files; examples: {examples}. "
            "Run the selective downloader before building a reportable manifest."
        )

    if args.skip_hand_validity_scan:
        validity_by_clip = {
            clip_id: {"left": (True,) * 150, "right": (True,) * 150}
            for clip_id, _, _, _, _ in clip_items
        }
    else:
        validity_by_clip: dict[int, dict[str, tuple[bool, ...]]] = {}
        with ThreadPoolExecutor(max_workers=args.scan_workers) as executor:
            futures = {
                executor.submit(_scan_one_clip, (clip_id, tar_path)): clip_id
                for clip_id, _, _, _, tar_path in clip_items
            }
            for completed, future in enumerate(as_completed(futures), start=1):
                clip_id, validity = future.result()
                validity_by_clip[clip_id] = validity
                if completed % 100 == 0 or completed == len(futures):
                    print(
                        json.dumps(
                            {
                                "phase": "hand_validity_scan",
                                "completed": completed,
                                "total": len(futures),
                            }
                        ),
                        flush=True,
                    )

    records_by_split: dict[str, list[dict[str, Any]]] = {
        split: [] for split in PARTICIPANT_SPLITS
    }
    counters: Counter[str] = Counter()
    for clip_id, definition, participant, split, tar_path in clip_items:
        sequence_id = str(definition["sequence_id"])
        validity = validity_by_clip[clip_id]
        for window_start in protocol.window_starts(150, stride=args.stride):
            temporal = protocol.manifest_fields(window_start)
            selected = tuple(temporal["history_indices"]) + tuple(temporal["future_indices"])
            for side in ("left", "right"):
                if not all(validity[side][frame] for frame in selected):
                    counters[f"{split}:incomplete_{side}_windows"] += 1
                    continue
                sample_id = f"hot3d:{clip_id:06d}:{window_start:03d}:{side}"
                records_by_split[split].append(
                    {
                        "schema_version": 1,
                        "protocol": "h6_k16_30hz",
                        "dataset": "hot3d_clips_aria",
                        "official_source_split": "train",
                        "split": split,
                        "sample_id": sample_id,
                        "source_group": participant,
                        "participant_id": participant,
                        "sequence_id": sequence_id,
                        "clip_id": clip_id,
                        "tar_path": str(tar_path.resolve()),
                        "tracked_hand": side,
                        "target": "future_wrist_translation_xyz",
                        "evaluation_target": "future_wrist_translation_xyz",
                        "target_coordinate_frame": "last_observed_camera",
                        "trajectory_window_start": window_start,
                        **temporal,
                    }
                )
                counters[f"{split}:{side}_windows"] += 1
        counters[f"{split}:clips"] += 1
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split, records in records_by_split.items():
        _write_jsonl(args.output_dir / f"hot3d_clips_aria_{split}_h6_k16.jsonl", records)

    participant_sets = {split: set(values) for split, values in PARTICIPANT_SPLITS.items()}
    if any(
        participant_sets[left] & participant_sets[right]
        for left in participant_sets
        for right in participant_sets
        if left < right
    ):
        raise AssertionError("HOT3D participant split leakage detected")
    summary = {
        "benchmark": "controlled_hot3d_clips_aria_h6_k16",
        "official_public_source_split": "train",
        "official_test_labels_used": False,
        "participant_splits": {
            split: list(values) for split, values in PARTICIPANT_SPLITS.items()
        },
        "records": {split: len(records) for split, records in records_by_split.items()},
        "missing_tar_count": len(missing),
        "strict_complete_22_frame_hand_tracks": not args.skip_hand_validity_scan,
        "counters": dict(sorted(counters.items())),
    }
    (args.output_dir / "hot3d_clips_aria_h6_k16_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
