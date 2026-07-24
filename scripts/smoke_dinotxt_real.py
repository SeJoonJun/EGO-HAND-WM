#!/usr/bin/env python3
"""Real-checkpoint smoke gate for the exact DINO.txt visual and language contracts."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch

from ego_hand_wm.data.dinov3_features import (
    LocalDinoTxtTextEncoder,
    LocalDinoTxtVisualEncoder,
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--rgb-root", type=Path, required=True)
    result.add_argument("--dataset", required=True)
    result.add_argument("--video", required=True)
    result.add_argument("--repo-path", type=Path, required=True)
    result.add_argument("--weights-path", type=Path, required=True)
    result.add_argument("--dinotxt-weights-path", type=Path, required=True)
    result.add_argument("--bpe-path", type=Path, required=True)
    result.add_argument("--frames", type=int, default=2)
    result.add_argument("--device", default="cuda")
    return result


def main() -> None:
    args = parser().parse_args()
    rgb_path = args.rgb_root / args.dataset / f"{args.video}.rgb.npy"
    rgb = np.load(rgb_path, allow_pickle=False, mmap_mode="r")
    frames = np.asarray(rgb[: args.frames])
    visual_encoder = LocalDinoTxtVisualEncoder(
        repo_path=args.repo_path,
        weights_path=args.weights_path,
        dinotxt_weights_path=args.dinotxt_weights_path,
        device=args.device,
    )
    visual = visual_encoder.encode(frames)
    if visual.shape != (args.frames, 17, 1024):
        raise ValueError(f"Unexpected real DINO.txt visual shape: {visual.shape}")
    global_descriptor = np.concatenate((visual[:, 0], visual[:, 1:].mean(axis=1)), axis=-1)
    global_descriptor /= np.linalg.norm(global_descriptor, axis=-1, keepdims=True)
    if global_descriptor.shape != (args.frames, 2048) or not np.isfinite(global_descriptor).all():
        raise ValueError("Invalid reconstructed DINO.txt global descriptors")
    del visual_encoder
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    text_encoder = LocalDinoTxtTextEncoder(
        repo_path=args.repo_path,
        dinotxt_weights_path=args.dinotxt_weights_path,
        bpe_path=args.bpe_path,
        device=args.device,
    )
    text = text_encoder.encode(["Right hand: pick up the cup", ""])
    if text.shape != (2, 2048) or not np.allclose(
        np.linalg.norm(text, axis=-1), 1.0, rtol=1e-4, atol=1e-4
    ):
        raise ValueError("Invalid normalized DINO.txt text descriptors")
    print(
        json.dumps(
            {
                "complete": True,
                "video": f"{args.dataset}/{args.video}",
                "visual_shape": list(visual.shape),
                "global_shape": list(global_descriptor.shape),
                "text_shape": list(text.shape),
                "visual_finite": bool(np.isfinite(visual).all()),
                "text_norms": np.linalg.norm(text, axis=-1).tolist(),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
