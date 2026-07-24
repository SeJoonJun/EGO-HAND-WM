#!/usr/bin/env python3
"""Verify USST H6/K16 loaders against the canonical trajectory manifests."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def _namespace(**kwargs):
    return SimpleNamespace(**kwargs)


def _records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _canonical_xyz(dataset, index: int) -> np.ndarray:
    item = dataset[index]
    return np.concatenate(
        (item["history_xyz_anchor"].numpy(), item["future_xyz_anchor"].numpy()),
        axis=0,
    )


def _check_manifest_contract(records: list[dict]) -> None:
    for record in records:
        assert record["protocol"] == "h6_k16_30hz"
        assert record["target_coordinate_frame"] == "last_observed_camera"
        assert float(record["fps"]) == 30.0
        assert int(record["history_steps"]) == 6
        assert int(record["future_steps"]) == 16
        assert len(record["history_indices"]) == 6
        assert len(record["future_indices"]) == 16
        assert int(record["anchor_index"]) == int(record["history_indices"][-1])


def _match_all(usst_items, canonical_items, *, tolerance: float) -> dict:
    if len(usst_items) != len(canonical_items):
        raise AssertionError(
            f"Window count mismatch: USST={len(usst_items)}, "
            f"canonical={len(canonical_items)}"
        )
    by_key: dict[tuple, list[tuple[str, np.ndarray]]] = defaultdict(list)
    for key, sample_id, xyz in canonical_items:
        by_key[key].append((sample_id, xyz))

    errors_all: list[float] = []
    mismatches: list[dict] = []
    for key, xyz in usst_items:
        candidates = by_key.get(key)
        if not candidates:
            raise AssertionError(f"USST window is absent from manifest: {key}")
        errors = [float(np.max(np.abs(xyz - candidate[1]))) for candidate in candidates]
        best = int(np.argmin(errors))
        error = errors[best]
        errors_all.append(error)
        if error > tolerance:
            mismatches.append(
                {
                    "key": repr(key),
                    "nearest": candidates[best][0],
                    "max_abs_error_m": error,
                }
            )
        candidates.pop(best)
        if not candidates:
            del by_key[key]
    if by_key:
        remaining = sum(len(value) for value in by_key.values())
        raise AssertionError(f"Canonical manifest has {remaining} unmatched windows")
    errors = np.asarray(errors_all, dtype=np.float64)
    return {
        "windows": len(usst_items),
        "mismatched_windows": len(mismatches),
        "max_abs_xyz_error_m": float(errors.max(initial=0.0)),
        "p50_abs_xyz_error_m": float(np.quantile(errors, 0.50)),
        "p95_abs_xyz_error_m": float(np.quantile(errors, 0.95)),
        "first_mismatch": mismatches[0] if mismatches else None,
    }


def _audit_h2o(root: Path, manifest: Path, phase: str, canonical_cls, module) -> dict:
    records = _records(manifest)
    _check_manifest_contract(records)
    canonical = canonical_cls(manifest, decode_rgb=False)
    data_cfg = _namespace(
        max_frames=22,
        history_steps=6,
        window_stride=16,
        include_tail_window=True,
        strict_fixed_windows=True,
        load_all=False,
    )
    model_cfg = _namespace(
        target="3d",
        modalities=["loc"],
        use_global=False,
        use_anchor=True,
        centralize=True,
        normalize=True,
    )
    dataset = module.H2O(
        str(root / "H2O"),
        phase=phase,
        transform=None,
        data_cfg=data_cfg,
        model_cfg=model_cfg,
    )
    canonical_items = []
    for index, record in enumerate(records):
        key = (
            str(Path(record["video_path"]).resolve()),
            int(record["history_indices"][0]),
            int(record["future_indices"][-1]),
        )
        canonical_items.append((key, record["sample_id"], _canonical_xyz(canonical, index)))
    usst_items = []
    for index in range(len(dataset)):
        video_path, _, _, _, trajectory = dataset[index]
        start, end = map(int, dataset.video_data["timestamps"][index])
        normalized = (trajectory.numpy() + 1.0) * 0.5
        xyz = module.denormalize_traj(normalized, target="3d", use_global=False)
        key = (str(Path(video_path).resolve()), start, end)
        usst_items.append((key, xyz.astype(np.float32)))
    return _match_all(usst_items, canonical_items, tolerance=2e-5)


def _audit_egopat(
    root: Path, manifest: Path, phase: str, canonical_cls, module
) -> dict:
    records = _records(manifest)
    _check_manifest_contract(records)
    canonical = canonical_cls(manifest, decode_rgb=False)
    data_cfg = _namespace(
        max_frames=22,
        history_steps=6,
        window_stride=16,
        include_tail_window=True,
        strict_fixed_windows=True,
        load_all=False,
        scenes=None,
        tinyset=False,
    )
    model_cfg = _namespace(
        target="3d",
        modalities=["loc"],
        use_odom=True,
        use_anchor=True,
        trajectory_normalization="xyz_standardize",
        xyz_mean=[0.05346673, 0.02785282, 0.52405508],
        xyz_std=[0.22611412, 0.14114624, 0.51840801],
        centralize=False,
    )
    dataset = module.EgoPAT3D(
        str(root / "EgoPAT3D"),
        phase=phase,
        transform=None,
        data_cfg=data_cfg,
        model_cfg=model_cfg,
    )
    canonical_items = []
    for index, record in enumerate(records):
        key = (
            str(Path(record["video_path"]).resolve()),
            int(record["trajectory_window_start"]),
        )
        canonical_items.append((key, record["sample_id"], _canonical_xyz(canonical, index)))
    usst_items = []
    raw_items = []
    for index in range(len(dataset)):
        video_path, _, _, _, trajectory = dataset[index]
        raw_xyz = dataset._get_anchor_traj3d(
            np.asarray(dataset.traj_data[index]["traj3d"]),
            np.asarray(dataset.odom_data[index]),
        )
        normalized = trajectory.numpy()
        xyz = (
            normalized * np.asarray(model_cfg.xyz_std, dtype=np.float32)
            + np.asarray(model_cfg.xyz_mean, dtype=np.float32)
        )
        key = (str(Path(video_path).resolve()), int(dataset.frame_offsets[index]))
        usst_items.append((key, xyz.astype(np.float32)))
        raw_items.append((key, raw_xyz.astype(np.float32)))
    return {
        "raw_anchor_geometry": _match_all(
            raw_items, canonical_items, tolerance=2e-5
        ),
        "model_target_roundtrip": _match_all(
            usst_items, canonical_items, tolerance=2e-5
        ),
    }


def _split_leakage(manifest_root: Path, dataset: str) -> dict:
    paths = sorted(manifest_root.glob(f"{dataset}_*_h6_k16.jsonl"))
    groups = {
        path.stem.removeprefix(f"{dataset}_").removesuffix("_h6_k16"): {
            record["source_group"] for record in _records(path)
        }
        for path in paths
    }
    overlaps = {}
    names = sorted(groups)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            count = len(groups[left] & groups[right])
            overlaps[f"{left}__{right}"] = count
            if count:
                raise AssertionError(
                    f"{dataset} source-group leakage between {left} and {right}: {count}"
                )
    return overlaps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/projects/torresani-lab/sejoon/datasets/USST/data"),
    )
    parser.add_argument(
        "--manifest-root",
        type=Path,
        default=Path("/home/jun.se/EGO-HAND-WM/data/h6_k16_manifests"),
    )
    parser.add_argument(
        "--usst-root",
        type=Path,
        default=Path("/home/jun.se/EGO-HAND-WM-REF/USST"),
    )
    parser.add_argument(
        "--dataset",
        choices=("all", "h2o", "egopat3d"),
        default="all",
        help="Run one dataset independently so failures cannot hide other results.",
    )
    args = parser.parse_args()

    sys.path.insert(0, str(args.usst_root))
    from src import EgoPAT3DLoader, H2OLoader
    from ego_hand_wm.benchmarks.trajectory_dataset import TrajectoryWindowDataset

    report = {}
    if args.dataset in ("all", "h2o"):
        report["h2o"] = {}
        for split in ("train", "val", "test"):
            report["h2o"][split] = _audit_h2o(
                args.data_root,
                args.manifest_root / f"h2o_{split}_h6_k16.jsonl",
                split,
                TrajectoryWindowDataset,
                H2OLoader,
            )
        report["h2o"]["source_group_overlap"] = _split_leakage(
            args.manifest_root, "h2o"
        )
    if args.dataset in ("all", "egopat3d"):
        report["egopat3d"] = {}
        for split, phase in (
            ("train", "train"),
            ("val", "val"),
            ("test_seen", "test"),
            ("test_novel", "test_novel"),
        ):
            report["egopat3d"][split] = _audit_egopat(
                args.data_root,
                args.manifest_root / f"egopat3d_{split}_h6_k16.jsonl",
                phase,
                TrajectoryWindowDataset,
                EgoPAT3DLoader,
            )
        report["egopat3d"]["source_group_overlap"] = _split_leakage(
            args.manifest_root, "egopat3d"
        )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
