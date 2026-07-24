#!/usr/bin/env python3
"""Benchmark real staged RGB throughput for safe DINO.txt extraction batch selection."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from ego_hand_wm.data.dinov3_features import LocalDinoTxtVisualEncoder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb-path", type=Path, required=True)
    parser.add_argument("--repo-path", type=Path, required=True)
    parser.add_argument("--weights-path", type=Path, required=True)
    parser.add_argument("--dinotxt-weights-path", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=4096)
    parser.add_argument("--batches", type=int, nargs="+", default=[512, 1024, 2048])
    args = parser.parse_args()
    rgb = np.load(args.rgb_path, allow_pickle=False, mmap_mode="r")
    frame_count = min(args.frames, len(rgb))
    encoder = LocalDinoTxtVisualEncoder(
        repo_path=args.repo_path,
        weights_path=args.weights_path,
        dinotxt_weights_path=args.dinotxt_weights_path,
        device="cuda",
    )
    encoder.encode(np.asarray(rgb[: min(8, frame_count)]))
    results = []
    for batch_size in args.batches:
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        try:
            for start in range(0, frame_count, batch_size):
                encoder.encode(np.asarray(rgb[start : min(start + batch_size, frame_count)]))
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            results.append(
                {
                    "batch_size": batch_size,
                    "frames": frame_count,
                    "seconds": elapsed,
                    "frames_per_second": frame_count / elapsed,
                    "peak_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
                    "status": "complete",
                }
            )
        except torch.OutOfMemoryError as error:
            torch.cuda.empty_cache()
            results.append(
                {"batch_size": batch_size, "status": "oom", "error": str(error)}
            )
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
