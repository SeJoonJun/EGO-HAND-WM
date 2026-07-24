#!/usr/bin/env python3
"""Benchmark frozen local DINOv3 spatial-feature extraction on one GPU."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA")
    sys.path.insert(0, str(args.repo))
    from dinov3.hub.backbones import dinov3_vitl16

    model = dinov3_vitl16(pretrained=False)
    state = torch.load(args.weights, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model = model.cuda().eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    results = []
    with torch.inference_mode():
        for batch_size in args.batch_sizes:
            images = torch.randn(
                batch_size,
                3,
                args.image_size,
                args.image_size,
                device="cuda",
                dtype=torch.float32,
            )
            torch.cuda.reset_peak_memory_stats()
            for _ in range(args.warmup):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    output = model.forward_features(images)["x_norm_patchtokens"]
            torch.cuda.synchronize()
            started = time.perf_counter()
            for _ in range(args.iterations):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    output = model.forward_features(images)["x_norm_patchtokens"]
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            frames = batch_size * args.iterations
            results.append(
                {
                    "batch_size": batch_size,
                    "iterations": args.iterations,
                    "seconds": elapsed,
                    "frames_per_second": frames / elapsed,
                    "milliseconds_per_frame": 1000.0 * elapsed / frames,
                    "latent_shape": list(output.shape),
                    "peak_gpu_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
                }
            )
            del images, output
            torch.cuda.empty_cache()

    print(
        json.dumps(
            {
                "gpu": torch.cuda.get_device_name(0),
                "torch": torch.__version__,
                "image_size": args.image_size,
                "dtype": "bfloat16",
                "results": results,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
