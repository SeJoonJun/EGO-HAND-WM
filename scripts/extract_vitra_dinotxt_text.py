#!/usr/bin/env python3
"""Collect all VITRA prompts in parallel, then encode each unique prompt with DINO.txt."""

from __future__ import annotations

import argparse
import glob
import io
import json
import tarfile
from pathlib import Path

import numpy as np

from ego_hand_wm.data.adapters.vitra import enumerate_vitra_prompts
from ego_hand_wm.data.dinotxt_text import write_text_feature_cache
from ego_hand_wm.data.dinov3_features import (
    LocalDinoTxtTextEncoder,
    build_dinotxt_extractor_metadata,
)
from ego_hand_wm.data.feature_shards import atomic_write_json


def collect(annotation_shards: list[Path], output_path: Path) -> dict[str, object]:
    prompts: set[str] = set()
    episodes = 0
    for shard in annotation_shards:
        with tarfile.open(shard, "r:*") as archive:
            for member in archive:
                if not member.isfile() or not member.name.endswith(".npy"):
                    continue
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise ValueError(f"Cannot read annotation member: {member.name}")
                episode = np.load(io.BytesIO(extracted.read()), allow_pickle=True).item()
                if not isinstance(episode, dict):
                    raise ValueError(f"Invalid VITRA episode: {member.name}")
                prompts.update(enumerate_vitra_prompts(episode))
                episodes += 1
    result = {
        "complete": True,
        "shards": [str(path) for path in annotation_shards],
        "episodes": episodes,
        "prompts": sorted(prompts),
    }
    atomic_write_json(output_path, result)
    return {**result, "prompts": len(prompts)}


def encode(args: argparse.Namespace) -> dict[str, object]:
    collection_paths = [Path(path) for path in sorted(glob.glob(args.collection_glob))]
    if not collection_paths:
        raise FileNotFoundError(f"No prompt collections match {args.collection_glob}")
    prompts: set[str] = set()
    episodes = 0
    for path in collection_paths:
        payload = json.loads(path.read_text())
        if payload.get("complete") is not True:
            raise ValueError(f"Incomplete prompt collection: {path}")
        prompts.update(str(text) for text in payload["prompts"])
        episodes += int(payload["episodes"])
    ordered = sorted(prompts)
    metadata = build_dinotxt_extractor_metadata(
        repo_path=args.repo_path,
        weights_path=args.weights_path,
        dinotxt_weights_path=args.dinotxt_weights_path,
        bpe_path=args.bpe_path,
        model_name="dinov3_vitl16",
        input_size=256,
        spatial_grid_size=4,
    )
    encoder = LocalDinoTxtTextEncoder(
        repo_path=args.repo_path,
        dinotxt_weights_path=args.dinotxt_weights_path,
        bpe_path=args.bpe_path,
        device=args.device,
    )
    chunks = []
    for start in range(0, len(ordered), args.batch_size):
        chunks.append(encoder.encode(ordered[start : start + args.batch_size]))
    features = np.concatenate(chunks, axis=0) if chunks else np.empty((0, 2048), np.float32)
    success = write_text_feature_cache(
        output_root=args.output_root,
        texts=ordered,
        features=features,
        metadata=metadata,
    )
    return {**success, "episodes_scanned": episodes, "collections": len(collection_paths)}


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    commands = root.add_subparsers(dest="command", required=True)
    collect_parser = commands.add_parser("collect")
    collect_parser.add_argument("--annotation-shard", type=Path, nargs="+", required=True)
    collect_parser.add_argument("--output-path", type=Path, required=True)
    encode_parser = commands.add_parser("encode")
    encode_parser.add_argument("--collection-glob", required=True)
    encode_parser.add_argument("--output-root", type=Path, required=True)
    encode_parser.add_argument("--repo-path", type=Path, required=True)
    encode_parser.add_argument("--weights-path", type=Path, required=True)
    encode_parser.add_argument("--dinotxt-weights-path", type=Path, required=True)
    encode_parser.add_argument("--bpe-path", type=Path, required=True)
    encode_parser.add_argument("--batch-size", type=int, default=256)
    encode_parser.add_argument("--device", default="cuda")
    return root


def main() -> None:
    args = parser().parse_args()
    if args.command == "collect":
        result = collect(args.annotation_shard, args.output_path)
    else:
        result = encode(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
