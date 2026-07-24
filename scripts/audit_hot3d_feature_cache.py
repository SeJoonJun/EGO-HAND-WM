#!/usr/bin/env python3
"""Validate every HOT3D DINO.txt cache gate against its source manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ego_hand_wm.benchmarks.hot3d_clips_dataset import (
    Hot3DClipsForecastDataset,
)
from ego_hand_wm.contracts import canonical_collate
from ego_hand_wm.data.trajectory_features import TrajectoryVisualFeatureStore


def _records(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _indices(length: int) -> tuple[int, ...]:
    return tuple(sorted({0, length // 2, length - 1}))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=Path("/home/jun.se/EGO-HAND-WM/data/hot3d_clips_h6_k16"),
    )
    parser.add_argument(
        "--feature-root",
        type=Path,
        default=Path("/scratch/jun.se/EGO-HAND-WM/TRAJECTORY_DINOTXT"),
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report: dict[str, object] = {"caches": {}, "extractor_ids": []}
    extractor_ids: set[str] = set()
    for split in ("train", "validation", "test"):
        manifest = (
            args.manifest_dir / f"hot3d_clips_aria_{split}_h6_k16.jsonl"
        )
        records = _records(manifest)
        expected_ids = [str(record["sample_id"]) for record in records]
        sequences = ("history", "future") if split != "test" else ("history",)
        for sequence in sequences:
            store = TrajectoryVisualFeatureStore(
                args.feature_root,
                dataset="hot3d_clips_aria",
                split=split,
                sequence=sequence,
                output_dtype=np.float16,
            )
            actual_ids = [str(value) for value in store.sample_ids]
            if actual_ids != expected_ids:
                raise AssertionError(
                    f"{split}/{sequence} sample IDs do not match manifest order"
                )
            selected = _indices(len(store.sample_ids))
            for index in selected:
                if not np.isfinite(store.features[index]).all():
                    raise ValueError(
                        f"{split}/{sequence}[{index}] contains non-finite values"
                    )
            extractor_ids.add(str(store.success["extractor_id"]))
            report["caches"][f"{split}/{sequence}"] = {
                "samples": len(store.sample_ids),
                "shape": list(store.features.shape),
                "dtype": str(store.features.dtype),
                "checked_indices": list(selected),
                "extractor_id": str(store.success["extractor_id"]),
            }

        dataset = Hot3DClipsForecastDataset(
            {
                "split": split,
                "manifest": str(manifest),
                "visual_feature_root": str(args.feature_root),
                "future_visual_feature_root": str(args.feature_root),
                "future_visual_splits": ["train", "validation"],
                "visual_feature_dtype": "float16",
                "missing_text_feature_dim": 2048,
                "decode_rgb": False,
            }
        )
        batch = canonical_collate([dataset[index] for index in _indices(len(dataset))])
        batch.validate()
        if batch.context_visual_features is None:
            raise AssertionError(f"{split} is missing history visual features")
        if split == "test" and batch.future_visual_latents is not None:
            raise AssertionError("Test split must not require future visual targets")
        if split != "test" and batch.future_visual_latents is None:
            raise AssertionError(f"{split} is missing future visual targets")

    if len(extractor_ids) != 1:
        raise AssertionError(f"Cache extractor mismatch: {sorted(extractor_ids)}")
    report["extractor_ids"] = sorted(extractor_ids)
    report["complete"] = True
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
