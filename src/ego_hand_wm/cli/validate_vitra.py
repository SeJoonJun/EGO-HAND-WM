from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from ego_hand_wm.data.adapters.vitra import canonicalize_vitra_episode, load_vitra_episode
from ego_hand_wm.geometry.se3 import pose9_to_matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("annotation")
    parser.add_argument("--history-steps", type=int, default=3)
    parser.add_argument("--future-steps", type=int, default=4)
    parser.add_argument("--fallback-fps", type=float, default=30.0)
    parser.add_argument("--left-mano-policy", choices=("mask", "mirror_x", "as_stored"), default="mask")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    episode = load_vitra_episode(args.annotation)
    needed = args.history_steps + args.future_steps
    if len(episode["extrinsics"]) < needed:
        raise ValueError(f"Episode has fewer than {needed} frames")
    history = list(range(args.history_steps))
    future = list(range(args.history_steps, needed))
    times = np.asarray(episode["video_decode_frame"], dtype=np.float64) / args.fallback_fps
    sample = canonicalize_vitra_episode(
        episode,
        history,
        future,
        times,
        left_mano_policy=args.left_mano_policy,
    )
    anchor_camera = pose9_to_matrix(sample["history_state"][-1, :9])
    identity_error = float((anchor_camera - torch.eye(4)).abs().max())
    if identity_error > 1e-4:
        raise AssertionError(f"Anchor camera is not identity; max error={identity_error}")
    print(
        json.dumps(
            {
                "status": "pass",
                "history_shape": list(sample["history_state"].shape),
                "future_shape": list(sample["future_state"].shape),
                "anchor_camera_identity_max_error": identity_error,
                "future_time_seconds": sample["future_time"].tolist(),
                "valid_stream_counts": sample["future_stream_mask"].sum(dim=0).tolist(),
                "metadata": sample["metadata"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

