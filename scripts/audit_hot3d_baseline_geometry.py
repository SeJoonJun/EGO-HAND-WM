#!/usr/bin/env python3
"""Verify that every adapted model consumes the same absolute HOT3D target."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch

from ego_hand_wm.benchmarks.hot3d_clips_dataset import (
    Hot3DClipsForecastDataset,
)
from ego_hand_wm.contracts.schema import SCHEMA


XYZ_MEAN = (
    0.2704353097511446,
    -0.06219469197952073,
    0.3184789241289701,
)
XYZ_STD = (
    0.11184064967022754,
    0.1866135967844919,
    0.10673271771077786,
)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _selected_indices(length: int) -> tuple[int, ...]:
    return tuple(sorted({0, length // 2, length - 1}))


def _canonical_xyz(sample: dict, tracked_hand: str) -> np.ndarray:
    state = torch.cat((sample["history_state"], sample["future_state"]), dim=0)
    wrist = SCHEMA.left_wrist if tracked_hand == "left" else SCHEMA.right_wrist
    return state[:, wrist.start : wrist.start + 3].cpu().numpy()


def _maximum_error(reference: np.ndarray, candidate: np.ndarray) -> float:
    return float(np.max(np.abs(reference.astype(np.float64) - candidate)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=Path("/home/jun.se/EGO-HAND-WM/data/hot3d_clips_h6_k16"),
    )
    parser.add_argument(
        "--mmtwin-root",
        type=Path,
        default=Path("/home/jun.se/EGO-HAND-WM-REF/MMTwin"),
    )
    parser.add_argument(
        "--handsonvlm-root",
        type=Path,
        default=Path("/home/jun.se/EGO-HAND-WM-REF/HandsOnVLM-release"),
    )
    parser.add_argument(
        "--usst-root",
        type=Path,
        default=Path("/home/jun.se/EGO-HAND-WM-REF/USST"),
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for repository in (args.mmtwin_root, args.handsonvlm_root, args.usst_root):
        repository_text = str(repository.resolve())
        if repository_text not in sys.path:
            sys.path.insert(0, repository_text)

    mmtwin = _load_module(
        "hot3d_mmtwin_loader",
        args.mmtwin_root / "data_utils/H6K16ManifestLoader.py",
    )
    handson = _load_module(
        "hot3d_handson_loader",
        args.handsonvlm_root / "handsonvlm/dataset/h6k16_dataset.py",
    )
    usst = _load_module(
        "hot3d_usst_loader",
        args.usst_root / "src/H6K16ManifestLoader.py",
    )

    report: dict[str, object] = {
        "coordinate_contract": "absolute_xyz_in_last_observed_camera_frame",
        "history_steps": 6,
        "future_steps": 16,
        "normalization": "train_split_xyz_standardization",
        "residual_to_last_observation": False,
        "splits": {},
    }
    for split in ("train", "validation", "test"):
        manifest = (
            args.manifest_dir / f"hot3d_clips_aria_{split}_h6_k16.jsonl"
        )
        canonical = Hot3DClipsForecastDataset(
            {"split": split, "manifest": str(manifest)}
        )
        mmtwin_dataset = mmtwin.H6K16ManifestDataset(
            manifest, decode_rgb=False
        )
        usst_dataset = usst.H6K16ManifestDataset(
            manifest,
            input_size=64,
            means=(0.5, 0.5, 0.5),
            stds=(0.5, 0.5, 0.5),
            xyz_mean=XYZ_MEAN,
            xyz_std=XYZ_STD,
        )
        maxima = {
            "mmtwin": 0.0,
            "handsonvlm": 0.0,
            "usst": 0.0,
        }
        checked = []
        for index in _selected_indices(len(canonical)):
            canonical_sample = canonical[index]
            record = canonical.records[index]
            reference = _canonical_xyz(
                canonical_sample, str(record["tracked_hand"])
            )
            mmtwin_xyz = mmtwin_dataset[index]["trajectory_anchor"].numpy()
            handson_xyz = handson._trajectory_anchor(record)
            _, _, _, _, usst_standardized = usst_dataset[index]
            usst_xyz = usst.metric_xyz(
                usst_standardized.numpy(), XYZ_MEAN, XYZ_STD
            )
            for name, candidate in (
                ("mmtwin", mmtwin_xyz),
                ("handsonvlm", handson_xyz),
                ("usst", usst_xyz),
            ):
                error = _maximum_error(reference, candidate)
                maxima[name] = max(maxima[name], error)
                if not np.allclose(
                    reference, candidate, rtol=1e-5, atol=1e-5
                ):
                    raise AssertionError(
                        f"{split}[{index}] {name} max error={error}"
                    )
            checked.append(index)
        report["splits"][split] = {
            "records": len(canonical),
            "checked_indices": checked,
            "max_abs_xyz_error_m": maxima,
        }

    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
