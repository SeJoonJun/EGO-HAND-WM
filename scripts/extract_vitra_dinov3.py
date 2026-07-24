#!/usr/bin/env python3
"""Extract or finalize shard-aligned spatial DINOv3 features for VITRA."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

from ego_hand_wm.data.dinov3_features import (
    LocalDinoV3SpatialEncoder,
    VideoResolver,
    build_extractor_metadata,
    extract_feature_shard,
)
from ego_hand_wm.data.feature_shards import finalize_feature_root


def _video_roots(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Video root must be DATASET=PATH, got {value!r}")
        dataset, path = value.split("=", 1)
        if not dataset or not path or dataset in result:
            raise ValueError(f"Invalid or duplicate video root: {value!r}")
        result[dataset] = Path(path)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract = subparsers.add_parser("extract", help="extract one annotation shard")
    extract.add_argument(
        "--annotation-shard",
        type=Path,
        nargs="+",
        required=True,
        help="One or more shards; the DINO model and video index are reused across all of them.",
    )
    extract.add_argument("--output-root", type=Path, required=True)
    extract.add_argument("--pts-root", type=Path, required=True)
    extract.add_argument("--repo-path", type=Path, required=True)
    extract.add_argument("--weights-path", type=Path, required=True)
    extract.add_argument("--model-name", default="dinov3_vitl16")
    extract.add_argument("--input-size", type=int, default=256)
    extract.add_argument("--spatial-grid-size", type=int, default=4)
    extract.add_argument("--batch-size", type=int, default=32)
    extract.add_argument("--device", default="auto")
    extract.add_argument("--video-root", action="append", default=[], metavar="DATASET=PATH")
    extract.add_argument("--force", action="store_true")

    finalize = subparsers.add_parser("finalize", help="publish _SUCCESS after every shard passes")
    finalize.add_argument("--annotation-glob", required=True)
    finalize.add_argument("--output-root", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "finalize":
        shards = [Path(path) for path in sorted(glob.glob(args.annotation_glob))]
        print(json.dumps(finalize_feature_root(shards, args.output_root), indent=2))
        return

    roots = _video_roots(args.video_root)
    metadata = build_extractor_metadata(
        repo_path=args.repo_path,
        weights_path=args.weights_path,
        model_name=args.model_name,
        input_size=args.input_size,
        spatial_grid_size=args.spatial_grid_size,
    )
    encoder = LocalDinoV3SpatialEncoder(
        repo_path=args.repo_path,
        weights_path=args.weights_path,
        model_name=args.model_name,
        input_size=args.input_size,
        spatial_grid_size=args.spatial_grid_size,
        device=args.device,
    )
    resolver = VideoResolver(roots)
    for annotation_shard in args.annotation_shard:
        manifest = extract_feature_shard(
            annotation_shard=annotation_shard,
            output_root=args.output_root,
            pts_root=args.pts_root,
            video_resolver=resolver,
            encoder=encoder,
            extractor_metadata=metadata,
            batch_size=args.batch_size,
            force=args.force,
        )
        print(json.dumps(manifest, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
