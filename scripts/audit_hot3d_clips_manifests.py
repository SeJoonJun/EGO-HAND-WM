#!/usr/bin/env python3
"""Audit controlled HOT3D-Clips manifests and decode real canonical samples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ego_hand_wm.benchmarks.hot3d_clips_dataset import Hot3DClipsForecastDataset
from ego_hand_wm.contracts import canonical_collate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records: dict[str, list[dict]] = {}
    report: dict[str, object] = {"splits": {}}
    for split in ("train", "validation", "test"):
        path = args.manifest_dir / f"hot3d_clips_aria_{split}_h6_k16.jsonl"
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        records[split] = rows
        dataset = Hot3DClipsForecastDataset({"split": split, "manifest": str(path)})
        indices = sorted({0, len(dataset) // 2, len(dataset) - 1})
        canonical_collate([dataset[index] for index in indices]).validate()
        report["splits"][split] = {
            "records": len(rows),
            "participants": sorted({row["participant_id"] for row in rows}),
            "clips": len({row["clip_id"] for row in rows}),
            "sequences": len({row["sequence_id"] for row in rows}),
            "decoded_sample_indices": indices,
        }

    for field in ("participant_id", "sequence_id", "clip_id", "sample_id"):
        sets = {
            split: {row[field] for row in rows} for split, rows in records.items()
        }
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
            overlap = sets[left] & sets[right]
            if overlap:
                raise ValueError(f"Cross-split {field} leakage between {left}/{right}: {overlap}")
    report["cross_split_leakage"] = 0
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
