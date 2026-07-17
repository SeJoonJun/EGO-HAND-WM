#!/usr/bin/env python3
"""Inspect one EgoVLA simulation HDF5 episode without changing it."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from ego_hand_wm.data.adapters.egovla_robot import (  # noqa: E402
    MissingManoAssetsError,
    RobotHDF5ContractError,
    inspect_egovla_robot_hdf5,
    require_mano_assets,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read and validate an EgoVLA robot HDF5 episode. No source array is modified and "
            "no MANO values are synthesized."
        )
    )
    parser.add_argument("episode", type=Path)
    parser.add_argument(
        "--allow-target-ee",
        action="store_true",
        help=(
            "Permit commanded target EE poses only when both realized EE arrays are absent. "
            "The JSON report records this non-equivalent provenance."
        ),
    )
    parser.add_argument(
        "--source-hz",
        type=float,
        default=30.0,
        help="Fallback control rate when the HDF5 file has no timestamps_s dataset (default: 30).",
    )
    parser.add_argument(
        "--mano-root",
        type=Path,
        help=(
            "Optional MANO root/model directory. Supplying it strictly requires non-empty "
            "MANO_LEFT.pkl and MANO_RIGHT.pkl; files are not unpickled."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        inspection = inspect_egovla_robot_hdf5(
            args.episode,
            allow_target_ee=args.allow_target_ee,
            source_hz=args.source_hz,
        )
        if args.mano_root is None:
            mano_status = {
                "ready": False,
                "reason": (
                    "not checked: pass --mano-root; full MANO export must remain disabled"
                ),
            }
        else:
            mano_status = require_mano_assets(args.mano_root).as_dict()
    except (
        FileNotFoundError,
        MissingManoAssetsError,
        RobotHDF5ContractError,
        RuntimeError,
        ValueError,
    ) as error:
        raise SystemExit(f"inspection failed: {error}") from error

    report = inspection.as_dict()
    report["source_path"] = str(args.episode.resolve())
    report["mano_assets"] = mano_status
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
