#!/usr/bin/env python3
"""Cross-check adapted baseline geometry against the canonical manifest loader."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

from ego_hand_wm.benchmarks.trajectory_dataset import TrajectoryWindowDataset


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def selected_indices(length: int) -> tuple[int, ...]:
    return tuple(sorted({0, length // 2, length - 1}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest-root",
        type=Path,
        default=Path("/home/jun.se/EGO-HAND-WM/data/h6_k16_manifests"),
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
    args = parser.parse_args()

    # The audit imports adapter modules directly from sibling repositories.
    # Add their repository roots so package-local imports resolve regardless of
    # the caller's working directory or PYTHONPATH.
    for repository in (args.mmtwin_root, args.handsonvlm_root):
        repository_text = str(repository.resolve())
        if repository_text not in sys.path:
            sys.path.insert(0, repository_text)

    mmtwin = load_module(
        "mmtwin_h6k16_loader",
        args.mmtwin_root / "data_utils/H6K16ManifestLoader.py",
    )
    handson = load_module(
        "handson_h6k16_loader",
        args.handsonvlm_root / "handsonvlm/dataset/h6k16_dataset.py",
    )

    report = {}
    manifests = sorted(args.manifest_root.glob("*_h6_k16.jsonl"))
    if not manifests:
        raise FileNotFoundError(args.manifest_root)
    for manifest in manifests:
        canonical = TrajectoryWindowDataset(manifest, decode_rgb=False)
        adapted_mmtwin = mmtwin.H6K16ManifestDataset(manifest, decode_rgb=False)
        maximum_error = 0.0
        checked = 0
        for index in selected_indices(len(canonical)):
            canonical_item = canonical[index]
            mmtwin_item = adapted_mmtwin[index]
            canonical_xyz = np.concatenate(
                (
                    canonical_item["history_xyz_anchor"].numpy(),
                    canonical_item["future_xyz_anchor"].numpy(),
                ),
                axis=0,
            )
            handson_xyz = handson._trajectory_anchor(canonical.records[index])
            mmtwin_xyz = mmtwin_item["trajectory_anchor"].numpy()
            for baseline, value in (
                ("mmtwin", mmtwin_xyz),
                ("handsonvlm", handson_xyz),
            ):
                error = float(np.max(np.abs(value - canonical_xyz)))
                maximum_error = max(maximum_error, error)
                if not np.allclose(value, canonical_xyz, rtol=1e-5, atol=1e-5):
                    raise AssertionError(
                        f"{manifest.name}[{index}] {baseline} max error={error}"
                    )
            checked += 1
        report[manifest.name] = {
            "records": len(canonical),
            "checked": checked,
            "max_abs_xyz_error_m": maximum_error,
        }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
