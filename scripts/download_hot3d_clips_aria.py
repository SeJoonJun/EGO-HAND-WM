#!/usr/bin/env python3
"""Download only the labeled official HOT3D-Clips Aria training package."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--local-dir",
        type=Path,
        required=True,
        help="Destination containing train_aria/ and the two official JSON manifests.",
    )
    parser.add_argument("--repo-id", default="bop-benchmark/hot3d")
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument(
        "--token-file",
        type=Path,
        help="Optional Hugging Face token file; its contents are never printed.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_workers < 1:
        raise ValueError("--max-workers must be positive")
    if args.token_file is not None:
        token = args.token_file.read_text(encoding="utf-8").strip()
        if not token:
            raise ValueError(f"Empty Hugging Face token file: {args.token_file}")
        os.environ["HF_TOKEN"] = token
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise RuntimeError(
            "Install huggingface_hub in the download environment before running this script"
        ) from error

    args.local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        local_dir=args.local_dir,
        allow_patterns=(
            "clip_splits.json",
            "clip_definitions.json",
            "train_aria/*.tar",
        ),
        max_workers=args.max_workers,
    )


if __name__ == "__main__":
    main()
